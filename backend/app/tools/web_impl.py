"""Self-contained web fetch / search implementation.

This single file is used in two places:

* imported by the backend (``local`` sandbox backend, unit tests);
* copied into the sandbox image as ``/usr/local/lib/a1_web.py`` and executed *inside the sandbox*
  (``python3 /usr/local/lib/a1_web.py fetch <url>``) so outbound web access is bounded by the
  sandbox network, never by the backend's network (which can see PostgreSQL & friends).

Only the standard library plus ``beautifulsoup4`` / ``html2text`` (optional) are required.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 DecillionAgent/1.0"
MAX_BYTES = 3_000_000


class WebError(Exception):
    pass


def check_url(url: str) -> str:
    """Reject non-http(s) URLs and anything resolving to a private / link-local address."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise WebError("Only absolute http(s) URLs are supported.")
    if os.environ.get("A1_ALLOW_PRIVATE_URLS") == "1":
        return url
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WebError(f"Cannot resolve host {parsed.hostname}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise WebError(f"Refusing to fetch {parsed.hostname}: it resolves to a private address ({ip}).")
    return url


class _NoPrivateRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(req: urllib.request.Request, timeout: int = 25):
    opener = urllib.request.build_opener(_NoPrivateRedirect)
    return opener.open(req, timeout=timeout)


def html_to_text(html: str, base_url: str = "") -> tuple[str, str]:
    """Return (title, markdown-ish text)."""
    title = ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "iframe"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        html = str(main)
    except Exception:  # noqa: BLE001
        pass
    try:
        import html2text

        h = html2text.HTML2Text()
        h.ignore_images = True
        h.body_width = 0
        h.baseurl = base_url
        text = h.handle(html)
    except Exception:  # noqa: BLE001
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def fetch_url(url: str, max_chars: int = 12000) -> dict:
    check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.5"})
    try:
        with _open(req) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_BYTES)
            final = resp.geturl()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise WebError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise WebError(f"Request failed: {getattr(exc, 'reason', exc)}") from exc
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype, re.I)
    if m:
        charset = m.group(1)
    body = raw.decode(charset, "replace")
    title = ""
    if "html" in ctype.lower() or body.lstrip().lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(body, final)
    else:
        text = body
    truncated = len(text) > max_chars
    return {"url": final, "status": status, "title": title, "content_type": ctype, "text": text[:max_chars], "truncated": truncated, "total_chars": len(text)}


def _ddg_unwrap(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        q = urllib.parse.parse_qs(parsed.query)
        if "uddg" in q:
            return q["uddg"][0]
    return href


def search_web(query: str, n: int = 6) -> list[dict]:
    searx = os.environ.get("SEARXNG_URL")
    if searx:
        url = searx.rstrip("/") + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in data.get("results", [])[:n]]

    from bs4 import BeautifulSoup

    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise WebError(f"Search failed: {getattr(exc, 'reason', exc)}") from exc
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for block in soup.select("div.result"):
        a = block.select_one("a.result__a")
        if not a:
            continue
        snip = block.select_one(".result__snippet")
        results.append({"title": a.get_text(" ", strip=True), "url": _ddg_unwrap(a.get("href", "")), "snippet": snip.get_text(" ", strip=True) if snip else ""})
        if len(results) >= n:
            break
    if not results:
        raise WebError("No results (the search provider may be rate limiting; try different wording or fetch a known URL).")
    return results


def format_results(results: list[dict]) -> str:
    return "\n\n".join(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1))


def _cli(argv: list[str]) -> int:
    try:
        if argv[0] == "fetch":
            out = fetch_url(argv[1], int(argv[2]) if len(argv) > 2 else 12000)
        elif argv[0] == "search":
            out = {"results": search_web(argv[1], int(argv[2]) if len(argv) > 2 else 6)}
        else:
            raise WebError("usage: a1_web.py fetch URL [max_chars] | search QUERY [n]")
        print(json.dumps(out, ensure_ascii=False))
        return 0
    except WebError as exc:
        print(json.dumps({"error": str(exc)}))
        return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
