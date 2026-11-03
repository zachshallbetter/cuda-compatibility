"""Write content/pages.json into the site: one excerpt and title set per page, and a page per research paper.

Usage: python3 scripts/build_site_meta.py [--check]
content/pages.json is the single source for every page's search title, share title, excerpt, eyebrow and
sheet chip. The excerpt is written once, for a quick read, and used everywhere a page is summarised: the meta
description, Open Graph and X descriptions, schema.org descriptions, the feed, llms.txt and the share card.
Research papers in docs/ are rendered to site/research/ with their own metadata, citation tags and card.
--check writes nothing and exits 1 when any generated file differs (make check runs it).
Standard library only, so CI can run it.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import posixpath
import re
import sys
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REGISTRY = ROOT / "content" / "pages.json"
ORIGIN = "https://cudacompat.com/"
REPO = "https://github.com/zachshallbetter/cuda-compatibility"
LICENSE = REPO + "/blob/main/LICENSE-MIT"


def load() -> dict:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for page in data["pages"]:
        page["url"] = ORIGIN + page["path"]
        page["image"] = ORIGIN + card_path(page)
        page["alt"] = (f"Share card: {page['share_title']}. {page['eyebrow']}, {page['chip'].title().replace(' / ', ' of ')}, "
                       "CUDA Compatibility, cudacompat.com, beside a wireframe of a GM200 graphics card")
    return data


def card_path(page: dict) -> str:
    if page["kind"] == "home":
        return "assets/social-card.png"
    if page["path"] == "docs/":
        return "assets/social/docs.png"
    return "assets/social/" + Path(page["path"]).stem + ".png"


# ---------- Markdown (the subset the research papers use) ----------

def _inline(text: str, link) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{escape(part[1:-1])}</code>")
            continue
        s = escape(part, quote=False).replace("&lt;br&gt;", "<br>")
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{escape(link(m[2]))}">{m[1]}</a>', s)
        s = re.sub(r'(?<!["=>])\b(https?://[^\s<)]+[^\s<).,;:])', lambda m: f'<a href="{m[1]}">{m[1]}</a>', s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-") or "section"


def markdown(md: str, link) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (h1 text, body html, [(id, h2 text)])."""
    lines = md.splitlines()
    html: list[str] = []
    toc: list[tuple[str, str]] = []
    ids: set[str] = set()
    h1 = ""
    last_heading = "Table"
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            lang = line[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            label = {"sh": "Shell commands", "text": "Output", "": "Code"}.get(lang, f"{lang} code")
            html.append(f'<pre aria-label="{label}"><code>{escape(chr(10).join(body))}</code></pre>')
            continue
        heading = re.match(r"(#{1,4})\s+(.*)", line)
        if heading:
            level, text = len(heading[1]), heading[2].strip()
            if level == 1:
                h1 = text
            else:
                anchor = slugify(text)
                while anchor in ids:
                    anchor += "-2"
                ids.add(anchor)
                inner = _inline(text, link)
                html.append(f'<h{level} id="{anchor}">{inner}</h{level}>')
                last_heading = re.sub(r"<[^>]+>", "", inner)
                if level == 2:
                    toc.append((anchor, last_heading))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            label = escape(last_heading, quote=True)
            table = [f'<div class="scroll" role="region" aria-label="{label} (scrolls sideways)" tabindex="0" style="overflow-x:auto">',
                     f'<table aria-label="{label}"><thead><tr>' + "".join(f'<th scope="col">{_inline(c, link)}</th>' for c in head) + "</tr></thead><tbody>"]
            for row in body:
                table.append("<tr>" + "".join(f"<td>{_inline(c, link)}</td>" for c in row) + "</tr>")
            table.append("</tbody></table></div>")
            html.append("".join(table))
            continue
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> "))
                i += 1
            html.append(f'<blockquote><p>{_inline(" ".join(quote), link)}</p></blockquote>')
            continue
        item = re.match(r"(\s*)([-*]|\d+\.)\s+(.*)", line)
        if item:
            ordered = item[2][0].isdigit()
            items = []
            while i < len(lines):
                m = re.match(r"(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                if m:
                    items.append(m[3])
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            html.append(f'<{tag} class="list">' + "".join(f"<li>{_inline(t, link)}</li>" for t in items) + f"</{tag}>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"(#{1,4}\s|```|\||>|\s*([-*]|\d+\.)\s)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        html.append(f"<p>{_inline(' '.join(para), link)}</p>")
    return h1, "\n".join(html), toc


# ---------- Pages ----------

def head_meta(page: dict, root: str, data: dict, extra: str = "") -> str:
    """The shared head for generated paper pages."""
    t, x = escape(page["share_title"]), escape(page["excerpt"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="only light"><meta name="theme-color" content="#F2F2EE"><meta name="description" content="{x}">
<link rel="icon" href="{root}favicon.ico" sizes="32x32">
<link rel="icon" type="image/svg+xml" href="{root}favicon.svg">
<link rel="apple-touch-icon" href="{root}apple-touch-icon.png">
<link rel="mask-icon" href="{root}mask-icon.svg" color="#000000">
<link rel="manifest" href="{root}manifest.webmanifest">
<link rel="alternate" type="application/atom+xml" title="CUDA Compatibility sheets" href="{root}feed.xml">
{extra}<link rel="author" href="{root}humans.txt">
<link rel="license" href="{LICENSE}">
<meta name="application-name" content="CUDA Compatibility">
<meta name="apple-mobile-web-app-title" content="CUDA Compat">
<meta name="format-detection" content="telephone=no">
<title>{escape(page["title"])} | CUDA Compatibility</title>
<link rel="canonical" href="{page["url"]}">
<meta name="author" content="Zach Shallbetter">
<meta name="robots" content="index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1">
<meta name="citation_title" content="{escape(page["h1"])}">
<meta name="citation_author" content="Shallbetter, Zach">
<meta name="citation_publication_date" content="{page["published"][:10].replace("-", "/")}">
<meta name="citation_online_date" content="{page["source_modified"][:10].replace("-", "/")}">
<meta name="citation_publisher" content="CUDA Compatibility">
<meta name="citation_public_url" content="{page["url"]}">
<meta property="og:type" content="article">
<meta property="article:published_time" content="{page["published"]}">
<meta property="article:modified_time" content="{page["source_modified"]}">
<meta property="article:author" content="https://zachshallbetter.com/">
<meta property="article:section" content="Research">
<meta property="og:site_name" content="CUDA Compatibility">
<meta property="og:url" content="{page["url"]}">
<meta property="og:locale" content="en_US">
<meta property="og:title" content="{t}">
<meta property="og:description" content="{x}">
<meta property="og:image" content="{page["image"]}">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{escape(page["alt"])}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{t}">
<meta name="twitter:description" content="{x}">
<meta name="twitter:image" content="{page["image"]}">
<meta name="twitter:image:alt" content="{escape(page["alt"])}">
<script type="application/ld+json">{json.dumps(paper_graph(page, data), ensure_ascii=False, separators=(",", ":"))}</script>
<link rel="stylesheet" href="{root}styles.css">
<link rel="stylesheet" href="{root}docs.css">
</head>
"""


def identity_nodes(data: dict) -> list[dict]:
    home = SITE / "index.html"
    graph = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', home.read_text(encoding="utf-8"), re.S)[1])["@graph"]
    keep = {"WebSite", "Person", "Organization"}
    nodes = [n for n in graph if n.get("@type") in keep]
    nodes.append({"@type": "SoftwareApplication", "@id": ORIGIN + "#software", "name": "CUDA Compatibility", "url": ORIGIN})
    return nodes


def paper_graph(page: dict, data: dict) -> dict:
    url, img = page["url"], page["image"]
    research = ORIGIN + "docs/research.html"
    return {"@context": "https://schema.org", "@graph": identity_nodes(data) + [
        {"@type": "WebPage", "@id": url, "url": url, "name": page["share_title"], "description": page["excerpt"],
         "inLanguage": "en", "isPartOf": {"@id": ORIGIN + "#website"}, "breadcrumb": {"@id": url + "#breadcrumb"},
         "primaryImageOfPage": {"@id": img}, "mainEntity": {"@id": url + "#article"},
         "datePublished": page["published"], "dateModified": page["source_modified"]},
        {"@type": "ImageObject", "@id": img, "url": img, "contentUrl": img, "width": 1200, "height": 630,
         "caption": page["alt"], "encodingFormat": "image/png"},
        {"@type": "ScholarlyArticle", "@id": url + "#article", "headline": page["h1"], "name": page["share_title"],
         "abstract": page["excerpt"], "description": page["excerpt"], "url": url, "mainEntityOfPage": {"@id": url},
         "image": {"@id": img}, "inLanguage": "en", "datePublished": page["published"], "dateModified": page["source_modified"],
         "author": {"@id": ORIGIN + "#author"}, "publisher": {"@id": ORIGIN + "#author"},
         "copyrightHolder": {"@id": ORIGIN + "#author"}, "copyrightYear": int(page["published"][:4]), "license": LICENSE,
         "isAccessibleForFree": True, "isPartOf": {"@id": ORIGIN + "#website"}, "about": {"@id": ORIGIN + "#software"},
         "isBasedOn": f"{REPO}/blob/main/{page['source']}", "articleSection": "Research", "position": int(page["chip"].split()[1]),
         "creativeWorkStatus": "Published", "genre": "Research record"},
        {"@type": "BreadcrumbList", "@id": url + "#breadcrumb", "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": ORIGIN},
            {"@type": "ListItem", "position": 2, "name": "Research and evidence", "item": research},
            {"@type": "ListItem", "position": 3, "name": page["share_title"], "item": url}]},
    ]}


def human_date(iso: str) -> str:
    y, m, d = iso[:10].split("-")
    months = "January February March April May June July August September October November December".split()
    return f"{int(d)} {months[int(m) - 1]} {y}"


def paper_page(page: dict, data: dict, papers: list[dict]) -> str:
    src = ROOT / page["source"]
    by_source = {p["source"]: p for p in papers}

    def link(target: str) -> str:
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return target
        path, _, frag = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page["source"]), path))
        if resolved in by_source:
            return Path(by_source[resolved]["path"]).name + (f"#{frag}" if frag else "")
        kind = "tree" if (ROOT / resolved).is_dir() else "blob"
        return f"{REPO}/{kind}/main/{resolved}" + (f"#{frag}" if frag else "")

    h1, body, toc = markdown(src.read_text(encoding="utf-8"), link)
    page["h1"] = h1
    index = papers.index(page)
    prev, nxt = (papers[index - 1] if index else None), (papers[index + 1] if index + 1 < len(papers) else None)
    rels = (f'<link rel="prev" href="{Path(prev["path"]).name}">\n' if prev else "") + \
           (f'<link rel="next" href="{Path(nxt["path"]).name}">\n' if nxt else "")
    template = (SITE / "docs" / "research.html").read_text(encoding="utf-8")
    top = template[template.index("<body>"):template.index("<main")]
    top = re.sub(r'href="(?!\.\./|https?:|#|mailto:)([^"]+)"', r'href="../docs/\1"', top)
    side = re.search(r'<nav aria-label="Documentation".*?</ul>', template, re.S)[0]
    side = side.replace(' class="current" aria-current="page"', "").replace(
        '<a href="../docs/research.html">', '<a class="current" aria-current="true" href="../docs/research.html">')
    foot = template[template.index('<footer class="foot">'):template.index("</footer>") + len("</footer>")]
    number = page["chip"].split()[1]
    foot = re.sub(r"<span>Sheet</span><span>[^<]*</span>", f"<span>Paper</span><span>{number} / {len(papers):02d}, {escape(page['share_title'])}</span>", foot)
    foot = re.sub(r"<span>Correct this sheet</span><span><a href=\"[^\"]*\">",
                  f'<span>Correct this paper</span><span><a href="{REPO}/issues/new?title=Correction%3A%20paper%20{number}%20{Path(page["path"]).stem}">', foot)
    foot = re.sub(r'<table class="rev".*?</table>',
                  '<table class="rev" aria-label="Revision table"><thead><tr><th scope="col">Rev</th><th scope="col">Date</th><th scope="col">Change</th><th scope="col">By</th></tr></thead><tbody>'
                  f'<tr><td>A</td><td>{page["published"][:10]}</td><td>First recorded</td><td>ZS</td></tr>'
                  + (f'<tr><td>B</td><td>{page["source_modified"][:10]}</td><td>Latest revision of the source record</td><td>ZS</td></tr>' if page["source_modified"][:10] != page["published"][:10] else "")
                  + '</tbody></table>', foot, flags=re.S)
    on_page = "".join(f'<li><a href="#{a}"><i>{chr(65 + k) if k < 26 else k}</i>{escape(t)}</a></li>' for k, (a, t) in enumerate(toc))
    current = ' class="current" aria-current="page"'
    others = "".join(f'<li><a{current if p is page else ""} href="{Path(p["path"]).name}"><i>{p["chip"].split()[1]}</i>{escape(p["share_title"])}</a></li>' for p in papers)

    def pager_link(p, word):
        return (f'<a rel="{word.lower()}" href="{Path(p["path"]).name}"><span class="sheet">{word} · paper {p["chip"].split()[1]}</span>'
                f'<span class="pager-title">{escape(p["share_title"])}</span></a>')
    pager = ('<nav class="pager" aria-label="Papers, previous and next">' + (pager_link(prev, "Previous") if prev else "<span></span>")
             + (pager_link(nxt, "Next") if nxt else '<a href="../docs/research.html"><span class="sheet">Back to</span><span class="pager-title">Research and evidence</span></a>') + "</nav>")
    cite = (f'Shallbetter, Z. ({page["published"][:4]}). {escape(page["h1"])}. CUDA Compatibility. {page["url"]}')
    return (head_meta(page, "../", data, rels) + top +
f"""<main id="main">
<article id="article" aria-labelledby="title">
<header class="band">
<div class="wrap" style="padding-top:56px;padding-bottom:56px;display:flex;flex-direction:column;gap:18px">
<nav aria-label="Breadcrumb" class="crumbs"><ol><li><a href="../index.html">Home</a></li><li><a href="../docs/research.html">Research and evidence</a></li><li aria-current="page">Paper {number} / {len(papers):02d}</li></ol></nav>
<p class="sheet" style="margin:0">{escape(page["eyebrow"])}</p>
<h1 id="title" class="disp" style="font-size:clamp(36px,4.2vw,58px);max-width:22ch">{escape(h1)}</h1>
<p class="excerpt" style="margin:0;font-size:20px;line-height:1.45;max-width:60ch;font-weight:500">{escape(page["excerpt"])}</p>
<p class="byline">By <a rel="author" href="https://zachshallbetter.com/">Zach Shallbetter</a> · Paper {number} of {len(papers):02d} · First recorded <time datetime="{page["published"][:10]}">{human_date(page["published"])}</time> · Updated <time datetime="{page["source_modified"][:10]}">{human_date(page["source_modified"])}</time></p>
</div>
</header>
<div class="wrap body" style="padding-top:56px;padding-bottom:96px">
{side}
<p class="sheet" style="margin:12px 0 0">Papers</p>
<ul class="side" style="border-top-width:1px">{others}</ul>
<p class="sheet" style="margin:12px 0 0">On this paper</p>
<ul class="side" style="border-top-width:1px">{on_page}</ul>
</nav>
<div class="prose" style="display:flex;flex-direction:column;gap:20px;min-width:0">
{body}
<aside class="note" aria-label="Source and citation"><p style="margin:0;font-size:15px">Source record: <a href="{REPO}/blob/main/{page["source"]}">{page["source"]}</a>. Cite as: {cite}. Project-maintained research, not peer reviewed or endorsed by NVIDIA.</p></aside>
{pager}
</div>
</div>
</article>
</main>

{foot}

</div>
<script src="../site.js" defer></script>
<script src="../analytics.js" defer></script>
</body>
</html>
""")


# ---------- Applying the registry to hand-written pages ----------

def _set_meta(s: str, attr: str, key: str, value: str) -> str:
    pattern = rf'(<meta {attr}="{re.escape(key)}" content=")[^"]*(")'
    new, n = re.subn(pattern, lambda m: m[1] + escape(value) + m[2], s, count=1)
    if n != 1:
        raise SystemExit(f"missing <meta {attr}=\"{key}\">")
    return new


def apply_meta(s: str, page: dict) -> str:
    title = page["title"] if page["kind"] == "home" else f'{page["title"]} | CUDA Compatibility'
    s = re.sub(r"<title>.*?</title>", f"<title>{escape(title)}</title>", s, count=1)
    s = _set_meta(s, "name", "description", page["excerpt"])
    for key in ("og:title", "og:description", "og:image:alt"):
        s = _set_meta(s, "property", key, {"og:title": page["share_title"], "og:description": page["excerpt"], "og:image:alt": page["alt"]}[key])
    for key in ("twitter:title", "twitter:description", "twitter:image:alt"):
        s = _set_meta(s, "name", key, {"twitter:title": page["share_title"], "twitter:description": page["excerpt"], "twitter:image:alt": page["alt"]}[key])
    match = re.search(r'(<script type="application/ld\+json">)(.*?)(</script>)', s, re.S)
    graph = json.loads(match[2])
    for node in graph["@graph"]:
        kind = node.get("@type")
        if kind == "WebPage":
            node["name"], node["description"] = page["share_title"], page["excerpt"]
        elif kind in ("TechArticle", "ScholarlyArticle"):
            node["name"], node["description"] = page["share_title"], page["excerpt"]
            if kind == "ScholarlyArticle":
                node["abstract"] = page["excerpt"]
        elif kind == "ImageObject" and node.get("@id") == page["image"]:
            node["caption"] = page["alt"]
    ld = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    return s[:match.start(2)] + ld + s[match.end(2):]


def link_papers_from_research_sheet(s: str, papers: list[dict]) -> str:
    for p in papers:
        s = s.replace(f'href="{REPO}/blob/main/{p["source"]}"', f'href="../{p["path"]}"')
    return s


def research_sheet_index(s: str, papers: list[dict]) -> str:
    """The research sheet lists every paper with its excerpt, between markers."""
    items = "".join(f'<li><a href="../{p["path"]}"><strong>{escape(p["share_title"])}</strong></a><br><span class="mute">{escape(p["excerpt"])}</span></li>' for p in papers)
    block = f'<!-- papers:start -->\n<ul class="list papers">{items}</ul>\n<!-- papers:end -->'
    new, n = re.subn(r"<!-- papers:start -->.*?<!-- papers:end -->", lambda _: block, s, flags=re.S)
    if n != 1:
        raise SystemExit("docs/research.html needs <!-- papers:start --> and <!-- papers:end --> markers")
    return new


# ---------- Site indexes ----------

def sitemap(pages: list[dict]) -> str:
    rows = "".join(f'  <url><loc>{p["url"]}</loc><lastmod>{p.get("source_modified", data_modified)[:10]}</lastmod>'
                   f'<image:image><image:loc>{p["image"]}</image:loc></image:image></url>\n' for p in pages)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
            'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">\n' + rows + "</urlset>\n")


def feed(pages: list[dict], published: dict[str, str]) -> str:
    newest = max(p.get("source_modified", data_modified) for p in pages)
    entries = "".join(f"""  <entry>
    <title>{xml_escape(p["share_title"])}</title>
    <link rel="alternate" type="text/html" href="{p["url"]}"/>
    <id>{p["url"]}</id>
    <published>{p.get("published") or published[p["path"]]}</published>
    <updated>{p.get("source_modified", data_modified)}</updated>
    <summary>{xml_escape(p["excerpt"])}</summary>
    <category term="{"Research" if p["kind"] == "paper" else "Documentation"}"/>
    <link rel="enclosure" type="image/png" href="{p["image"]}"/>
  </entry>
""" for p in pages)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">
  <title>CUDA Compatibility</title>
  <subtitle>Sheets and research papers on running selected CUDA 13 PyTorch operations on older NVIDIA GPUs.</subtitle>
  <link rel="self" type="application/atom+xml" href="{ORIGIN}feed.xml"/>
  <link rel="alternate" type="text/html" href="{ORIGIN}"/>
  <id>{ORIGIN}</id>
  <updated>{newest}</updated>
  <author><name>Zach Shallbetter</name><uri>https://zachshallbetter.com/</uri></author>
  <rights>Text MIT or Apache-2.0, at your option. Copyright 2026 Zach Shallbetter.</rights>
  <icon>{ORIGIN}icon-192.png</icon>
  <logo>{ORIGIN}icon-512.png</logo>
{entries}</feed>
"""


def llms(s: str, papers: list[dict]) -> str:
    block = "## Research papers\n" + "".join(f'- [{p["share_title"]}]({p["url"]}): {p["excerpt"]}\n' for p in papers) + "\n"
    if "## Research papers\n" in s:
        return re.sub(r"## Research papers\n.*?\n\n", lambda _: block, s, count=1, flags=re.S)
    return s.replace("## Machine-readable files\n", block + "## Machine-readable files\n", 1)


data_modified = ""


def build() -> dict[Path, str]:
    """Every generated or updated file and its exact new text."""
    global data_modified
    data = load()
    data_modified = data["modified"]
    pages = data["pages"]
    papers = [p for p in pages if p["kind"] == "paper"]
    out: dict[Path, str] = {}
    for p in papers:
        out[SITE / p["file"]] = paper_page(p, data, papers)
    published: dict[str, str] = {}
    for p in pages:
        if p["kind"] == "paper":
            continue
        path = SITE / p["file"]
        s = path.read_text(encoding="utf-8")
        if p["path"] == "docs/research.html":
            s = research_sheet_index(link_papers_from_research_sheet(s, papers), papers)
        s = apply_meta(s, p)
        s = re.sub(r'<meta property="article:modified_time" content="[^"]*">', f'<meta property="article:modified_time" content="{data_modified}">', s)
        found = re.search(r'<meta property="article:published_time" content="([^"]+)"', s)
        published[p["path"]] = found[1] if found else "2026-11-01T01:04:31-08:00"
        out[path] = s
    out[SITE / "sitemap.xml"] = sitemap(pages)
    out[SITE / "feed.xml"] = feed(pages, published)
    out[SITE / "llms.txt"] = llms((SITE / "llms.txt").read_text(encoding="utf-8"), papers)
    return out


def problems() -> list[str]:
    """Registry rules plus freshness of every generated file."""
    errors = []
    data = load()
    seen = set()
    for p in data["pages"]:
        if not 140 <= len(p["excerpt"]) <= 158:
            errors.append(f'{p["path"] or "/"}: excerpt is {len(p["excerpt"])} characters; write 140–158 so it fits search and shares')
        if len(p["title"]) > 60:
            errors.append(f'{p["path"] or "/"}: search title is over 60 characters')
        if len(p["share_title"]) > 70:
            errors.append(f'{p["path"] or "/"}: share title is over 70 characters')
        if p["excerpt"] in seen:
            errors.append(f'{p["path"] or "/"}: excerpt duplicates another page')
        seen.add(p["excerpt"])
    for path, text in build().items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            errors.append(f"{path.relative_to(ROOT)} is stale; run python3 scripts/build_site_meta.py")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = problems()
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print("OK page registry: excerpts, titles and generated pages are current" if not errors else f"{len(errors)} registry problem(s)")
        return 1 if errors else 0
    (SITE / "research").mkdir(exist_ok=True)
    for path, text in build().items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
