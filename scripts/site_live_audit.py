"""Audit the public site against this checkout after a deploy.

Usage: python3 scripts/site_live_audit.py [--origin URL] [--json]
Every canonical page and site file must return 200 with the right type and serve exactly the committed bytes;
each page's canonical, og:url and share image must resolve; security.txt must not have expired.
Exit 0 when everything matches, 1 otherwise. Read-only: it sends GET requests and nothing else.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ORIGIN = "https://cudacompat.com/"

# Site files beside the pages: path -> expected Content-Type prefix.
FILES = {
    "robots.txt": "text/plain",
    "sitemap.xml": "application/xml",
    "llms.txt": "text/plain",
    "llms-full.txt": "text/plain",
    "feed.xml": "application/atom+xml",
    "humans.txt": "text/plain",
    ".well-known/security.txt": "text/plain",
    "manifest.webmanifest": "application/manifest+json",
    "favicon.ico": "image/",
    "favicon.svg": "image/svg+xml",
    "apple-touch-icon.png": "image/png",
    "03dd79a90f0607e442d7c5e89bd241ee.txt": "text/plain",
}

Fetch = Callable[[str], tuple[int, str, bytes]]


def fetch(url: str) -> tuple[int, str, bytes]:
    request = Request(f"{url}{'&' if '?' in url else '?'}audit={int(time.time())}", headers={"Cache-Control": "no-cache"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()
    except URLError:
        return 0, "", b""


def page_file(url: str, origin: str) -> Path:
    rel = url[len(origin):]
    return SITE / (rel + "index.html" if rel == "" or rel.endswith("/") else rel)


def audit(get: Fetch = fetch, origin: str = ORIGIN, site: Path = SITE, now: datetime | None = None) -> list[str]:
    errors: list[str] = []
    pages = [n.text or "" for n in ElementTree.parse(site / "sitemap.xml").getroot().findall("{*}url/{*}loc")]
    images: set[str] = set()
    for url in pages:
        status, ctype, body = get(url)
        local = (site / page_file(url, pages[0]).relative_to(SITE)).read_bytes()
        if status != 200 or not ctype.startswith("text/html"):
            errors.append(f"{url}: HTTP {status} {ctype}")
            continue
        if body != local:
            errors.append(f"{url}: live bytes differ from the checkout (deploy not landed, or a stale checkout)")
        text = body.decode("utf-8", "replace")
        if f'<link rel="canonical" href="{url}">' not in text or f'<meta property="og:url" content="{url}">' not in text:
            errors.append(f"{url}: canonical or og:url is not the page's own URL")
        image = re.search(r'<meta property="og:image" content="([^"]+)"', text)
        if image:
            images.add(image[1])
    for image in sorted(images):
        status, ctype, _ = get(image)
        if status != 200 or ctype != "image/png":
            errors.append(f"{image}: HTTP {status} {ctype}")
    for rel, want in FILES.items():
        status, ctype, body = get(origin + rel)
        if status != 200 or not ctype.startswith(want):
            errors.append(f"/{rel}: HTTP {status} {ctype}, expected {want}")
        elif body != (site / rel).read_bytes():
            errors.append(f"/{rel}: live bytes differ from the checkout")
    status, _, body = get(origin + "no-such-sheet-" + str(int(time.time())))
    if status != 404 or b'name="robots" content="noindex' not in body:
        errors.append(f"unknown paths should return the noindex 404 page, got HTTP {status}")
    expires = re.search(rb"(?m)^Expires: (\S+)", (site / ".well-known/security.txt").read_bytes())
    if not expires:
        errors.append("security.txt has no Expires field")
    else:
        when = datetime.fromisoformat(expires[1].decode().replace("Z", "+00:00"))
        if when <= (now or datetime.now(timezone.utc)):
            errors.append("security.txt has expired; move Expires forward")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--origin", default=ORIGIN)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    errors = audit(origin=args.origin)
    if args.json:
        print(json.dumps({"origin": args.origin, "ok": not errors, "errors": errors}, indent=2))
    else:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print("OK live site matches the checkout" if not errors else f"{len(errors)} live problem(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
