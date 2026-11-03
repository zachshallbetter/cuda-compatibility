"""Submit canonical URLs to IndexNow after a publish (Bing and other participants).

Usage: python3 scripts/indexnow.py [--since REV] [--wait-live SECONDS] [--dry-run]
With --since, only pages whose files changed between REV and HEAD are sent; with no such page, nothing is sent.
With --wait-live, each page (and the key file) must serve exactly the committed bytes before anything is sent,
so a submission never points crawlers at a deploy that has not landed.
IndexNow is a change notification, not an indexing guarantee, and Google does not participate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ENDPOINT = "https://api.indexnow.org/indexnow"


def sitemap_urls(site: Path = SITE) -> list[str]:
    return [node.text or "" for node in ElementTree.parse(site / "sitemap.xml").getroot().findall(".//{*}loc")]


def key_file(site: Path = SITE) -> Path:
    keys = [p for p in site.glob("*.txt") if len(p.stem) == 32 and p.read_text(encoding="utf-8").strip() == p.stem]
    if len(keys) != 1:
        raise SystemExit("expected exactly one IndexNow key file in site/")
    return keys[0]


def page_file(url: str, origin: str, site: Path = SITE) -> Path:
    """The committed file a canonical URL serves: a trailing slash means that directory's index.html."""
    rel = url[len(origin):]
    return site / (rel + "index.html" if rel == "" or rel.endswith("/") else rel)


def changed_urls(paths: list[str], urls: list[str], site: Path = SITE) -> list[str]:
    """Sitemap URLs whose page file is among the changed repository paths (shared assets alone send nothing)."""
    changed = {(ROOT / p).resolve() for p in paths}
    return [u for u in urls if page_file(u, urls[0], site).resolve() in changed]


def git_changed(rev: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", rev, "HEAD", "--", "site"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def payload(urls: list[str], site: Path = SITE) -> dict:
    key = key_file(site)
    origin = sitemap_urls(site)[0]
    return {"host": urlsplit(origin).netloc, "key": key.stem, "keyLocation": f"{origin}{key.name}", "urlList": urls}


def fetch(url: str) -> bytes:
    with urlopen(Request(f"{url}?indexnow={int(time.time())}", headers={"Cache-Control": "no-cache"}), timeout=30) as r:
        return r.read()


def wait_live(files: dict[str, Path], seconds: float, interval: float = 15) -> bool:
    """True once every URL serves its committed bytes; False if that hasn't happened within `seconds`."""
    pending = dict(files)
    deadline = time.monotonic() + seconds
    while True:
        for url, path in list(pending.items()):
            try:
                if fetch(url) == path.read_bytes():
                    del pending[url]
            except OSError:
                pass
        if not pending:
            return True
        if time.monotonic() >= deadline:
            print("not live yet: " + ", ".join(pending), file=sys.stderr)
            return False
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", help="send only pages changed between this revision and HEAD")
    parser.add_argument("--wait-live", type=float, metavar="SECONDS", help="wait until the deploy serves these pages")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    urls = sitemap_urls()
    if args.since:
        urls = changed_urls(git_changed(args.since), urls)
        if not urls:
            print("IndexNow: no canonical page changed; nothing to send")
            return 0
    body = payload(urls)
    if args.dry_run:
        print(json.dumps(body, indent=2))
        return 0
    if args.wait_live is not None:
        files = {u: page_file(u, sitemap_urls()[0]) for u in urls}
        files[body["keyLocation"]] = key_file()
        if not wait_live(files, args.wait_live):
            return 1
    request = Request(ENDPOINT, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    with urlopen(request, timeout=30) as response:
        print(f"IndexNow responded {response.status} for {len(urls)} URLs: " + " ".join(urls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
