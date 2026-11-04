#!/usr/bin/env python3
"""Check that no page scrolls sideways on a 390 px phone screen.

Usage: python3 scripts/check_mobile_width.py [--chrome PATH] [--width N] [page.html ...]

Each page is loaded in headless Chrome inside an iframe of the given width (Chrome will not size a headless
window below 500 px), and the document's scrollWidth is compared with its clientWidth. Content inside scroll
regions and code blocks may scroll; the page itself may not. Needs Chrome, so it runs locally, not in CI.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

WRAPPER = """<!doctype html><body style="margin:0"><iframe id="f" src="{src}" style="width:{width}px;height:900px;border:0">
</iframe><script>f.onload = () => {{ const d = f.contentDocument.documentElement;
document.title = "W=" + d.scrollWidth + "/" + d.clientWidth; }};</script></body>"""


def pages() -> list[Path]:
    return sorted(SITE.rglob("*.html"))


def measure(chrome: str, page: Path, width: int) -> tuple[int, int]:
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=page.parent, prefix=".width-", delete=False) as f:
        f.write(WRAPPER.format(src=page.name, width=width))
        wrapper = Path(f.name)
    try:
        out = subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--allow-file-access-from-files",
                              "--window-size=800,1000", "--virtual-time-budget=4000", "--dump-dom", wrapper.as_uri()],
                             capture_output=True, text=True, timeout=60).stdout
    finally:
        wrapper.unlink()
    match = re.search(r"<title>W=(\d+)/(\d+)</title>", out)
    if not match:
        raise RuntimeError(f"{page.relative_to(SITE)}: Chrome returned no measurement")
    return int(match[1]), int(match[2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chrome", default=CHROME)
    parser.add_argument("--width", type=int, default=390)
    parser.add_argument("pages", nargs="*", type=Path)
    args = parser.parse_args(argv)
    targets = [p.resolve() for p in args.pages] or pages()
    wide = []
    for page in targets:
        scroll, client = measure(args.chrome, page, args.width)
        if scroll > client:
            wide.append(f"{page.relative_to(SITE)}: {scroll - client} px wider than a {args.width} px screen")
    for line in wide:
        print(f"WIDE {line}")
    if wide:
        return 1
    print(f"OK {len(targets)} pages fit a {args.width} px screen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
