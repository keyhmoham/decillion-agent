from pathlib import Path

import httpx
import pytest

from app.tools.search import (
    BingProvider, BraveProvider, DuckDuckGoProvider, SearchError, SearchService, SearxngProvider, build_providers,
    decode_bing_url, parse_bing, parse_duckduckgo,
)

FIX = Path(__file__).parent / "fixtures"


def test_bing_parser_decodes_redirects_from_a_real_page():
    res = parse_bing((FIX / "bing_results.html").read_text(), 5)
    assert len(res) == 3
    assert all(r.url.startswith("http") and "bing.com/ck" not in r.url for r in res)
    assert res[0].title and res[0].snippet


def test_decode_bing_url_roundtrip_and_passthrough():
    import base64
    real = "https://realpython.com/async-io-python/"
    href = "https://www.bing.com/ck/a?!&&p=x&u=a1" + base64.urlsafe_b64encode(real.encode()).decode().rstrip("=") + "&ntb=1"
    assert decode_bing_url(href) == real
    assert decode_bing_url("https://example.com/x") == "https://example.com/x"


def test_duckduckgo_parser_unwraps_uddg():
    html = ('<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=1">Title A</a>'
            '<a class="result__snippet">Snippet A</a></div>')
    r = parse_duckduckgo(html, 3)
    assert r[0].url == "https://example.com/a" and r[0].title == "Title A" and r[0].snippet == "Snippet A"


def client_for(handler):
    return httpx.MockTransport(handler)


async def run_provider(provider, handler, q="python"):
    async with httpx.AsyncClient(transport=client_for(handler)) as c:
        return await provider.search(c, q, 5)


async def test_ddg_bot_challenge_is_reported_as_rate_limit():
    html = (FIX / "ddg_anomaly.html").read_text()
    with pytest.raises(SearchError) as e:
        await run_provider(DuckDuckGoProvider(), lambda r: httpx.Response(202, text=html))
    assert e.value.rate_limited and "bot challenge" in str(e.value)


async def test_searxng_json_and_failure_modes():
    ok = lambda r: httpx.Response(200, json={"results": [{"title": "T", "url": "https://a.test", "content": "C"}]})  # noqa: E731
    res = await run_provider(SearxngProvider("http://searxng:8080"), ok)
    assert res[0].url == "https://a.test" and res[0].snippet == "C"
    with pytest.raises(SearchError, match="JSON format disabled"):
        await run_provider(SearxngProvider("http://s"), lambda r: httpx.Response(403))
    with pytest.raises(SearchError, match="no results"):
        await run_provider(SearxngProvider("http://s"), lambda r: httpx.Response(200, json={"results": []}))


async def test_brave_sends_token_and_parses():
    seen = {}

    def handler(r):
        seen["tok"] = r.headers["x-subscription-token"]
        return httpx.Response(200, json={"web": {"results": [{"title": "B", "url": "https://b.test", "description": "D"}]}})

    res = await run_provider(BraveProvider("KEY"), handler)
    assert seen["tok"] == "KEY" and res[0].snippet == "D"


class Fake:
    min_interval = 0.0

    def __init__(self, name, results=None, error=None):
        self.name, self.results, self.error, self.calls = name, results, error, 0

    async def search(self, client, query, n):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


async def test_service_falls_through_cools_down_and_caches():
    from app.tools.search import SearchResult

    limited = Fake("ddg", error=SearchError("rate limited", rate_limited=True))
    good = Fake("bing", results=[SearchResult("T", "https://x.test", "s")])
    svc = SearchService([limited, good], cooldown=60)
    res, who = await svc.search("hello world")
    assert who == "bing" and res[0].url == "https://x.test" and limited.calls == 1
    res, who = await svc.search("another query")  # cooled-down provider is skipped, not hammered
    assert who == "bing" and limited.calls == 1
    res, who = await svc.search("hello world")
    assert who.endswith("(cached)") and good.calls == 2
    assert any(s["provider"] == "ddg" and s["cooling_down"] for s in svc.status())


async def test_transient_error_is_retried_once_and_total_failure_is_actionable():
    from app.tools.search import SearchResult

    flaky_calls = {"n": 0}

    class Flaky(Fake):
        async def search(self, client, query, n):
            flaky_calls["n"] += 1
            if flaky_calls["n"] == 1:
                raise SearchError("timeout")
            return [SearchResult("T", "https://x.test")]

    _, who = await SearchService([Flaky("a")]).search("q1")
    assert who == "a" and flaky_calls["n"] == 2
    dead = SearchService([Fake("a", error=SearchError("boom")), Fake("b", error=SearchError("limited", rate_limited=True))])
    with pytest.raises(SearchError) as e:
        await dead.search("q2")
    assert "a: boom" in str(e.value) and "b: limited" in str(e.value) and "SEARXNG_URL" in str(e.value)


def test_provider_chain_construction():
    names = [p.name for p in build_providers("searxng,brave,bing,duckduckgo", "http://s:8080", "")]
    assert names == ["searxng", "bing", "duckduckgo"]  # brave skipped without a key
    assert [p.name for p in build_providers("brave", "", "")] == ["bing", "duckduckgo"]  # never an empty chain


async def test_web_search_tool_returns_results_and_actionable_errors(tmp_path):
    from app.tools.registry import build_tools
    from app.tools.search import SearchResult

    from .helpers import make_ctx

    ctx, *_ = make_ctx(tmp_path)
    ctx.search = SearchService([Fake("fake", results=[SearchResult("Hello", "https://h.test", "snip")])])
    tool = {t.name: t for t in build_tools(ctx, ("web",))}["web_search"]
    out = await tool.ainvoke({"query": "hello"})
    assert "Hello" in out and "https://h.test" in out and "[via fake]" in out
    ctx.search = SearchService([Fake("x", error=SearchError("limited", rate_limited=True))])
    assert (await tool.ainvoke({"query": "hello"})).startswith("Error: All search providers failed")


@pytest.mark.skipif(not __import__("os").environ.get("LIVE_SEARCH"), reason="set LIVE_SEARCH=1 to hit the real network")
async def test_live_bing_provider():
    res, who = await SearchService([BingProvider()]).search("python asyncio tutorial", 5)
    assert who == "bing" and len(res) >= 3 and all(r.url.startswith("http") for r in res)


async def test_web_fetch_refuses_private_targets_and_cli_has_no_search(tmp_path):
    from app.tools import web_impl
    from app.tools.registry import build_tools

    from .helpers import make_ctx

    ctx, *_ = make_ctx(tmp_path)
    tools = {t.name: t for t in build_tools(ctx, ("web",))}
    assert set(tools) == {"web_search", "web_fetch"}
    out = await tools["web_fetch"].ainvoke({"url": "http://127.0.0.1:9/secret"})
    assert out.startswith("Error:") and "private address" in out
    assert not hasattr(web_impl, "search_web")  # single-engine scraper removed (search lives in tools/search.py)


@pytest.mark.skipif(not __import__("os").environ.get("LIVE_SEARCH"), reason="set LIVE_SEARCH=1 to hit the real network")
async def test_live_web_fetch(tmp_path):
    from app.tools.registry import build_tools

    from .helpers import make_ctx

    ctx, *_ = make_ctx(tmp_path)
    out = await {t.name: t for t in build_tools(ctx, ("web",))}["web_fetch"].ainvoke({"url": "https://example.com"})
    assert "Example Domain" in out
