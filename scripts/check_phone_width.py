"""Render every page at phone width in headless Chrome and report any that scroll sideways.

Usage: python3 scripts/check_phone_width.py [--chrome PATH] [--width 390] [page ...]
Each page is copied into a scratch copy of site/ with a probe script and loaded in an iframe of the
given width (headless Chrome will not open a window narrower than 500 px), in the light theme and
then the dark one. A page fails when it is wider than the frame or a literal Markdown link (`](`)
is visible. Exit 0 when every page fits, 1 otherwise.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROBE = """<script>addEventListener("load", () => setTimeout(() => {
  const root = document.documentElement, wide = [];
  for (const el of document.body.querySelectorAll("*")) {
    const box = el.getBoundingClientRect();
    if (box.right > innerWidth + 1 && !el.closest("pre,.scroll,table,[style*=overflow]")) wide.push(el.tagName.toLowerCase());
  }
  const literal = document.body.innerText.includes("](");
  parent.postMessage(`PROBE ${root.scrollWidth} ${innerWidth} ${literal ? 1 : 0} ${wide.slice(0, 3).join(",")}`, "*");
}, 300));</script>"""


def pages() -> list[Path]:
    return sorted(p for p in SITE.rglob("*.html") if "node_modules" not in p.parts and p.name != "404.html")


FRAME = """<!doctype html><title>waiting</title><iframe src="{src}" width="{width}" height="844" style="border:0"></iframe>
<script>addEventListener("message", event => {{ document.title = event.data; }});</script>"""


def measure(chrome: str, copy: Path, rel: Path, width: int, scheme: str) -> str | None:
    page = copy / f"frame-{scheme}-{str(rel).replace('/', '-')}"
    page.write_text(FRAME.format(src=rel.as_posix(), width=width), encoding="utf-8")
    with tempfile.TemporaryDirectory() as profile:
        args = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                "--allow-file-access-from-files", f"--user-data-dir={profile}", "--window-size=1024,900",
                "--virtual-time-budget=4000", "--dump-dom", page.as_uri()]
        if scheme == "dark":
            args.insert(1, "--force-dark-mode")
            args.insert(1, "--blink-settings=preferredColorScheme=0")
        for _ in range(3):
            done = subprocess.run(args, capture_output=True, text=True, timeout=120)
            found = re.search(r"<title>PROBE (\d+) (\d+) (\d) ?([^<]*)</title>", done.stdout)
            if found:
                break
        else:
            return f"{rel} ({scheme}): Chrome did not render it"
    scroll, viewport, literal, wide = int(found[1]), int(found[2]), found[3] == "1", found[4]
    problems = []
    if scroll > viewport:
        problems.append(f"{scroll - viewport} px wider than {viewport} px" + (f" ({wide})" if wide else ""))
    if literal:
        problems.append("shows a literal Markdown link")
    return f"{rel} ({scheme}): " + "; ".join(problems) if problems else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chrome", default=CHROME)
    parser.add_argument("--width", type=int, default=390)
    parser.add_argument("pages", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if not Path(args.chrome).exists():
        print(f"skipped: no Chrome at {args.chrome} (pass --chrome)", file=sys.stderr)
        return 0
    targets = [p.resolve().relative_to(SITE) for p in args.pages] if args.pages else [p.relative_to(SITE) for p in pages()]
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "site"
        shutil.copytree(SITE, copy)
        for rel in targets:
            path = copy / rel
            path.write_text(path.read_text(encoding="utf-8").replace("</body>", PROBE + "</body>", 1), encoding="utf-8")
        jobs = [(rel, scheme) for rel in targets for scheme in ("light", "dark")]
        with ThreadPoolExecutor(max_workers=6) as pool:
            problems = [p for p in pool.map(lambda job: measure(args.chrome, copy, job[0], args.width, job[1]), jobs) if p]
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    print(f"OK {len(targets)} pages fit {args.width} px in light and dark" if not problems
          else f"{len(problems)} page view(s) do not fit {args.width} px")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
