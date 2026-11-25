"""Resilient web search.

Scraping a single search engine fails in practice (DuckDuckGo answers datacenter IPs with an HTTP 202
"anomaly" challenge; engines throttle bursts).  This module therefore:

* tries several providers in order - self-hosted **SearXNG** (shipped in docker-compose, aggregates many
  engines), **Brave Search API** (when ``BRAVE_API_KEY`` is set), **Bing** and **DuckDuckGo** HTML;
* puts a provider on a short **cooldown** after it rate-limits/blocks us, so later calls skip it;
* spaces requests per provider (**throttle**), retries transient errors once, and caches answers;
* returns an actionable message when everything fails.

It runs in the backend (not in the sandbox) because the query text is the only thing sent out.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class SearchError(Exception):
    def __init__(self, message: str, *, rate_limited: bool = False) -> None:
        super().__init__(message)
        self.rate_limited = rate_limited


class Provider(Protocol):
    name: str
    min_interval: float

    async def search(self, client: httpx.AsyncClient, query: str, n: int) -> list[SearchResult]: ...


def _clean(s: str) -> str:
    return " ".join((s or "").split())


# --------------------------------------------------------------------------------------- providers
class SearxngProvider:
    name = "searxng"
    min_interval = 0.2

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    async def search(self, client: httpx.AsyncClient, query: str, n: int) -> list[SearchResult]:
        try:
            r = await client.get(f"{self.base}/search", params={"q": query, "format": "json", "language": "en"})
        except httpx.HTTPError as exc:
            raise SearchError(f"searxng unreachable at {self.base}: {exc!r}") from exc
        if r.status_code == 403:
            raise SearchError("searxng refused the request (JSON format disabled? see docker/searxng/settings.yml)")
        if r.status_code == 429:
            raise SearchError("searxng rate limited", rate_limited=True)
        if r.status_code >= 400:
            raise SearchError(f"searxng returned HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError as exc:
            raise SearchError("searxng returned a non-JSON response") from exc
        out = [SearchResult(_clean(x.get("title", "")), x.get("url", ""), _clean(x.get("content", ""))) for x in data.get("results", []) if x.get("url")]
        if not out:
            raise SearchError("searxng returned no results (its upstream engines may be throttled)")
        return out[:n]


class BraveProvider:
    name = "brave"
    min_interval = 1.1  # free tier: 1 request/second

    def __init__(self, api_key: str) -> None:
        self.key = api_key

    async def search(self, client: httpx.AsyncClient, query: str, n: int) -> list[SearchResult]:
        try:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": min(n, 20)},
                headers={"X-Subscription-Token": self.key, "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise SearchError(f"brave unreachable: {exc!r}") from exc
        if r.status_code in (401, 403):
            raise SearchError("brave rejected the API key")
        if r.status_code == 429:
            raise SearchError("brave rate limited", rate_limited=True)
        if r.status_code >= 400:
            raise SearchError(f"brave returned HTTP {r.status_code}")
        items = (r.json().get("web") or {}).get("results", [])
        out = [SearchResult(_clean(x.get("title", "")), x.get("url", ""), _clean(x.get("description", ""))) for x in items if x.get("url")]
        if not out:
            raise SearchError("brave returned no results")
        return out[:n]


def decode_bing_url(href: str) -> str:
    """Bing wraps results in ``/ck/a?...&u=a1<base64url(real url)>``."""
    try:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        u = q.get("u", [""])[0]
        if u.startswith("a1"):
            b = u[2:]
            return base64.urlsafe_b64decode(b + "=" * (-len(b) % 4)).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        pass
    return href


def parse_bing(html: str, n: int) -> list[SearchResult]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[SearchResult] = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a")
        if not a or not a.get("href"):
            continue
        p = li.select_one(".b_caption p") or li.select_one("p")
        url = decode_bing_url(a["href"])
        if url.startswith("http"):
            out.append(SearchResult(_clean(a.get_text(" ")), url, _clean(p.get_text(" ")) if p else ""))
        if len(out) >= n:
            break
    return out


class BingProvider:
    name = "bing"
    min_interval = 1.5

    async def search(self, client: httpx.AsyncClient, query: str, n: int) -> list[SearchResult]:
        try:
            r = await client.get("https://www.bing.com/search", params={"q": query, "setlang": "en", "count": min(n, 20)},
                                 headers={"Accept-Language": "en-US,en;q=0.9"})
        except httpx.HTTPError as exc:
            raise SearchError(f"bing unreachable: {exc!r}") from exc
        if r.status_code in (403, 429):
            raise SearchError("bing blocked the request", rate_limited=True)
        if r.status_code >= 400:
            raise SearchError(f"bing returned HTTP {r.status_code}")
        out = parse_bing(r.text, n)
        if not out:
            raise SearchError("bing returned no parsable results (captcha or layout change)", rate_limited="captcha" in r.text.lower())
        return out


def parse_duckduckgo(html: str, n: int) -> list[SearchResult]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[SearchResult] = []
    for block in soup.select("div.result"):
        a = block.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
        snip = block.select_one(".result__snippet")
        out.append(SearchResult(_clean(a.get_text(" ")), href, _clean(snip.get_text(" ")) if snip else ""))
        if len(out) >= n:
            break
    return out


class DuckDuckGoProvider:
    name = "duckduckgo"
    min_interval = 2.5

    async def search(self, client: httpx.AsyncClient, query: str, n: int) -> list[SearchResult]:
        try:
            r = await client.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "wt-wt"})
        except httpx.HTTPError as exc:
            raise SearchError(f"duckduckgo unreachable: {exc!r}") from exc
        if r.status_code == 202 or "anomaly" in r.text[:6000].lower():
            raise SearchError("duckduckgo is rate limiting this IP (bot challenge)", rate_limited=True)
        if r.status_code >= 400:
            raise SearchError(f"duckduckgo returned HTTP {r.status_code}", rate_limited=r.status_code == 429)
        out = parse_duckduckgo(r.text, n)
        if not out:
            raise SearchError("duckduckgo returned no results")
        return out


# ------------------------------------------------------------------------------------------ service
@dataclass
class SearchService:
    providers: list[Provider]
    cache_ttl: float = 600.0
    cooldown: float = 90.0
    transport: httpx.AsyncBaseTransport | None = None  # tests
    _cache: OrderedDict = field(default_factory=OrderedDict)
    _cool_until: dict[str, float] = field(default_factory=dict)
    _last_call: dict[str, float] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    async def _throttle(self, p: Provider) -> None:
        lock = self._locks.setdefault(p.name, asyncio.Lock())
        async with lock:
            wait = self._last_call.get(p.name, 0.0) + p.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call[p.name] = time.monotonic()

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [{"provider": p.name, "cooling_down": self._cool_until.get(p.name, 0) > now,
                 "retry_in": max(0, round(self._cool_until.get(p.name, 0) - now))} for p in self.providers]

    async def search(self, query: str, n: int = 6) -> tuple[list[SearchResult], str]:
        """Return (results, provider_name). Raises SearchError with a combined, actionable message."""
        query = _clean(query)
        if not query:
            raise SearchError("empty query")
        key = (query.lower(), n)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl:
            self._cache.move_to_end(key)
            return hit[1], hit[2] + " (cached)"

        failures: list[str] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0), follow_redirects=True,
                                     headers={"User-Agent": UA}, transport=self.transport) as client:
            for p in self.providers:
                if self._cool_until.get(p.name, 0) > time.monotonic():
                    failures.append(f"{p.name}: cooling down after rate limit")
                    continue
                for attempt in (1, 2):
                    await self._throttle(p)
                    try:
                        results = await p.search(client, query, n)
                    except SearchError as exc:
                        if exc.rate_limited:
                            self._cool_until[p.name] = time.monotonic() + self.cooldown
                            failures.append(f"{p.name}: {exc}")
                            break
                        if attempt == 2:
                            failures.append(f"{p.name}: {exc}")
                        continue  # one retry for transient errors
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{p.name}: {type(exc).__name__}: {exc}")
                        break
                    self._cache[key] = (time.monotonic(), results, p.name)
                    while len(self._cache) > 256:
                        self._cache.popitem(last=False)
                    return results, p.name
        raise SearchError(
            "All search providers failed (" + "; ".join(failures) + "). Try different wording, or use web_fetch on a URL you already know. "
            "Operators: set SEARXNG_URL (bundled in docker-compose) or BRAVE_API_KEY for reliable search."
        )


def build_providers(names: str, searxng_url: str, brave_key: str) -> list[Provider]:
    avail: dict[str, Provider | None] = {
        "searxng": SearxngProvider(searxng_url) if searxng_url else None,
        "brave": BraveProvider(brave_key) if brave_key else None,
        "bing": BingProvider(),
        "duckduckgo": DuckDuckGoProvider(),
    }
    chosen = [avail[n.strip()] for n in names.split(",") if n.strip() in avail and avail[n.strip()] is not None]
    return [p for p in chosen if p is not None] or [BingProvider(), DuckDuckGoProvider()]


def format_results(results: list[SearchResult], provider: str) -> str:
    body = "\n\n".join(f"{i}. {r.title}\n   {r.url}\n   {r.snippet}" for i, r in enumerate(results, 1))
    return f"{body}\n\n[via {provider}]"
