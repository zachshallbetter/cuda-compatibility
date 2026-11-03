"""Submit the sitemap's canonical URLs to IndexNow after a meaningful publish (Bing and other participants).

Usage: python3 scripts/indexnow.py [--dry-run]
IndexNow is a change notification, not an indexing guarantee, and Google does not participate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

SITE = Path(__file__).resolve().parents[1] / "site"


def payload() -> dict:
    urls = [node.text or "" for node in ElementTree.parse(SITE / "sitemap.xml").getroot().findall(".//{*}loc")]
    keys = [p for p in SITE.glob("*.txt") if len(p.stem) == 32 and p.read_text(encoding="utf-8").strip() == p.stem]
    if len(keys) != 1:
        raise SystemExit("expected exactly one IndexNow key file in site/")
    origin = urls[0]
    return {
        "host": urlsplit(origin).netloc,
        "key": keys[0].stem,
        "keyLocation": f"{origin}{keys[0].name}",
        "urlList": urls,
    }


def main() -> int:
    body = payload()
    if "--dry-run" in sys.argv:
        print(json.dumps(body, indent=2))
        return 0
    request = Request(
        "https://api.indexnow.org/indexnow",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        print(f"IndexNow responded {response.status} for {len(body['urlList'])} URLs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
