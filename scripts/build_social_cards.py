"""Render the 1200x630 share cards and the no-JavaScript hero wireframe with headless Chrome.

Usage: python3 scripts/build_social_cards.py [--chrome PATH]
Each card takes its title, summary and sheet from the page it represents, so a card shared on its own says
what the page is. Cards are ink on paper, matching the site; the wireframe is drawn from the same data the
home page animates (site/assets/gm200-wireframe.json), settled and unlabelled.
"""
from __future__ import annotations

import argparse
from html import escape, unescape
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# (page, card file, eyebrow); the title, summary and sheet number come from the page itself.
CARDS = [
    ("index.html", "assets/social-card.png", "Independent compatibility research"),
    ("docs/index.html", "assets/social/docs.png", "Documentation"),
    ("docs/getting-started.html", "assets/social/getting-started.png", "Documentation"),
    ("docs/compatibility.html", "assets/social/compatibility.png", "Documentation"),
    ("docs/architecture.html", "assets/social/architecture.png", "Documentation"),
    ("docs/cli.html", "assets/social/cli.png", "Documentation"),
    ("docs/research.html", "assets/social/research.png", "Documentation"),
    ("gpu-catalog.html", "assets/social/gpu-catalog.png", "Hardware reference"),
    ("docs/discovery.html", "assets/social/discovery.png", "Documentation"),
    ("docs/privacy.html", "assets/social/privacy.png", "Documentation"),
]

# The settled wireframe, drawn like site.js: same projection, depth-banded ink, no fan spin, no labels.
WIREFRAME_JS = """
async function wireframe(canvas, ox, oy, scale, alpha) {
  const d = await (await fetch(WIRE)).json();
  const { q, n, e } = d, ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  const yaw = 0.62, pitch = -0.5, f = 3.2, k = scale, buckets = 7, P = new Float32Array(n * 6);
  const rot = (x, y, z, o) => {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x2 = cy * x + sy * z, z2 = -sy * x + cy * z;
    P[o] = x2; P[o + 1] = cp * y - sp * z2; P[o + 2] = sp * y + cp * z2;
  };
  for (let i = 0; i < n; i++) {
    const b = i * 6;
    rot(e[b] / q, e[b + 1] / q, e[b + 2] / q, b);
    rot(e[b + 3] / q, e[b + 4] / q, e[b + 5] / q, b + 3);
  }
  const paths = Array.from({ length: buckets }, () => new Path2D());
  const X = (x, z) => w * ox + x * (k / (f + z)), Y = (y, z) => h * oy - y * (k / (f + z));
  for (let j = 0; j < n; j++) {
    const z = (P[j * 6 + 2] + P[j * 6 + 5]) * 0.5, depth = Math.max(0, Math.min(1, (z + 1.1) / 2.2));
    const p = paths[Math.min(buckets - 1, Math.floor((1 - depth) * buckets))];
    p.moveTo(X(P[j * 6], P[j * 6 + 2]), Y(P[j * 6 + 1], P[j * 6 + 2]));
    p.lineTo(X(P[j * 6 + 3], P[j * 6 + 5]), Y(P[j * 6 + 4], P[j * 6 + 5]));
  }
  ctx.lineWidth = canvas.dataset.line || 1;
  for (let bk = 0; bk < buckets; bk++) {
    ctx.strokeStyle = `rgba(0,0,0,${((0.14 + (bk / (buckets - 1)) * 0.5) * alpha).toFixed(3)})`;
    ctx.stroke(paths[bk]);
  }
}
"""

FONTS = """
@font-face{font-family:"Archivo";font-weight:400 900;font-stretch:62% 125%;src:url(SITE/assets/fonts/archivo-latin.woff2) format("woff2")}
@font-face{font-family:"Martian Mono";font-weight:400 500;src:url(SITE/assets/fonts/martian-mono-latin.woff2) format("woff2")}
"""

MARK = ('<svg class="reg" style="{pos}" viewBox="0 0 22 22" fill="none" stroke="#000" stroke-width="1">'
        '<path d="M11 0v22M0 11h22"/><circle cx="11" cy="11" r="6"/></svg>')

CARD = """<!doctype html><html><head><meta charset="utf-8"><style>
FONTS
*{box-sizing:border-box;margin:0}
html,body{width:1200px;height:630px;overflow:hidden;background:#F2F2EE;color:#111111;font-family:"Archivo",Helvetica,Arial,sans-serif}
canvas{position:absolute;inset:0}
.reg{position:absolute;width:22px;height:22px}
.top{position:absolute;left:64px;right:64px;top:52px;display:flex;justify-content:space-between;align-items:center}
.brand{display:flex;align-items:center;gap:14px;font-weight:800;font-stretch:110%;font-size:25px;letter-spacing:-0.02em}
.chip{background:#FFD400;color:#000;border:2px solid #000;padding:7px 12px;font:500 15px "Martian Mono",monospace;letter-spacing:0.06em}
.body{position:absolute;left:64px;top:150px;width:720px;display:flex;flex-direction:column;gap:22px}
.eyebrow{font:600 17px "Archivo",sans-serif;letter-spacing:0.12em;text-transform:uppercase;color:#4A4A46}
h1{font-weight:800;font-stretch:112%;letter-spacing:-0.035em;line-height:0.98;color:#000}
h1 .ink{color:#1F3DFF}
.sum{font-size:25px;line-height:1.36;color:#3A3A36;max-width:720px}
.foot{position:absolute;left:64px;right:64px;bottom:52px;border-top:3px solid #000;padding-top:18px;display:flex;justify-content:space-between;align-items:center}
.url{font:500 19px "Martian Mono",monospace;color:#000}
.tag{font-weight:800;font-size:17px;letter-spacing:0.08em;text-transform:uppercase}
</style></head><body>
<canvas id="c" width="1200" height="630"></canvas>
MARKS
<div class="top"><div class="brand"><svg width="30" height="30" viewBox="0 0 32 32" fill="none" stroke="#000" stroke-width="3"><rect x="5" y="5" width="22" height="22"/><rect x="12" y="12" width="8" height="8" fill="#000" stroke="none"/></svg>CUDA Compatibility</div><span class="chip">SHEET</span></div>
<div class="body"><p class="eyebrow">EYEBROW</p><h1 id="t">TITLE</h1><p class="sum">SUMMARY</p></div>
<div class="foot"><span class="url">URL</span><span class="tag">TAG</span></div>
<script>
const WIRE = "WIREURL";
WIREFRAME_JS
const t = document.getElementById("t");
document.fonts.ready.then(async () => {
  let size = 84;
  t.style.fontSize = size + "px";
  while (t.getBoundingClientRect().height > 250 && size > 44) { size -= 2; t.style.fontSize = size + "px"; }
  await wireframe(document.getElementById("c"), 0.87, 0.56, 1450, 0.55);
  document.title = "ready";
});
</script></body></html>"""

HERO = """<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:transparent}</style></head>
<body><canvas id="c" width="2400" height="1500" data-line="2"></canvas><script>
const WIRE = "WIREURL";
WIREFRAME_JS
wireframe(document.getElementById("c"), 0.5, 0.5, 4200, 1).then(() => document.title = "ready");
</script></body></html>"""


def page_facts(rel: str) -> dict:
    s = (SITE / rel).read_text(encoding="utf-8")
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", s, re.S)[1]
    title = unescape(re.sub(r"<[^>]+>", "", h1)).replace("­", "").strip()
    canonical = re.search(r'<link rel="canonical" href="([^"]+)"', s)[1]
    sheet = re.search(r"<span>Sheet</span><span>(\d\d) / (\d\d)", s)
    summary = unescape(re.search(r'<meta property="og:description" content="([^"]*)"', s)[1])
    return {"title": title, "url": canonical.removeprefix("https://"), "summary": summary,
            "sheet": f"SHEET {sheet[1]} / {sheet[2]}"}


def card_html(rel: str, eyebrow: str) -> str:
    facts = page_facts(rel)
    title = escape(facts["title"])
    if rel == "index.html":
        title = 'Newer <span class="ink">CUDA.</span><br>Older <span class="ink">GPUs.</span>'
    marks = "".join(MARK.format(pos=p) for p in ("top:16px;left:16px", "top:16px;right:16px",
                                                 "bottom:16px;left:16px", "bottom:16px;right:16px"))
    # WIREURL goes first: the later "URL" placeholder is a substring of it.
    out = CARD.replace("WIREURL", (SITE / "assets/gm200-wireframe.json").as_uri())
    for key, value in (("FONTS", FONTS.replace("SITE/", SITE.as_uri() + "/")), ("MARKS", marks),
                       ("WIREFRAME_JS", WIREFRAME_JS), ("SHEET", facts["sheet"]), ("EYEBROW", escape(eyebrow)),
                       ("TITLE", title), ("SUMMARY", escape(facts["summary"])), ("URL", escape(facts["url"])),
                       ("TAG", "Independent. Not affiliated with NVIDIA." if rel == "index.html" else "Newer CUDA. Older GPUs.")):
        out = out.replace(key, value)
    return out


def shoot(chrome: str, html: str, out: Path, size: tuple[int, int], transparent: bool = False) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "card.html"
        page.write_text(html, encoding="utf-8")
        shot = Path(tmp) / "shot.png"
        args = [chrome, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                "--allow-file-access-from-files", f"--user-data-dir={tmp}/profile", "--force-device-scale-factor=1",
                f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=8000", f"--screenshot={shot}"]
        if transparent:
            args.append("--default-background-color=00000000")
        subprocess.run(args + [page.as_uri()], check=True, capture_output=True, timeout=120)
        shutil.copyfile(shot, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chrome", default=CHROME)
    args = parser.parse_args()
    for rel, card, eyebrow in CARDS:
        shoot(args.chrome, card_html(rel, eyebrow), SITE / card, (1200, 630))
        print(f"wrote site/{card}")
    hero = HERO.replace("WIREFRAME_JS", WIREFRAME_JS).replace("WIREURL", (SITE / "assets/gm200-wireframe.json").as_uri())
    out = SITE / "assets/gm200-wireframe.png"
    shoot(args.chrome, hero, out, (2400, 1500), transparent=True)
    from PIL import Image  # only this step needs Pillow: crop to the drawing and store ink as grey + alpha

    alpha = Image.open(out).getchannel("A")
    x0, y0, x1, y1 = alpha.getbbox()
    Image.merge("LA", (Image.new("L", alpha.size, 0), alpha)).crop((x0 - 24, y0 - 24, x1 + 24, y1 + 24)).save(out, optimize=True)
    print("wrote site/assets/gm200-wireframe.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
