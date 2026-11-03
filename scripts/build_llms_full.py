"""Build site/llms-full.txt: the main text of every sitemap page as plain Markdown."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from xml.etree import ElementTree


HEADER = """# CUDA Compatibility — full text

> The readable text of every page listed in {sitemap}, in sitemap order, generated from the published HTML. Each page's canonical URL precedes its text. This is a convenience copy for people and automated agents; the HTML pages are the source of truth, and this file is not a crawler directive.

A successful CUDA version query, import, or profile audit does not prove workload compatibility. Do not infer support for unlisted operators, APIs, libraries, data types, devices, training, or performance.
"""

SKIP = {"script", "style", "svg", "canvas", "nav", "button"}
BOUNDARY = "\x00"
INLINE = {"span", "a", "code", "strong", "b", "em", "i", "small", "kbd"}
BLOCK = {"p", "div", "section", "article", "ul", "ol", "table", "dl", "pre", "figure", "header", "footer"}


class MainText(HTMLParser):
    """Collects the text inside <main>, rendering headings, lists, tables and code as Markdown."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.in_main = 0
        self.skip = 0
        self.pre = 0
        self.cells: list[str] | None = None
        self.cell: list[str] | None = None
        self.header_row = False
        self.rows_emitted = 0
        self.skip_tags: list[str] = []

    def _emit(self, text: str) -> None:
        if self.cell is not None:
            self.cell.append(text)
        else:
            self.out.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "main":
            self.in_main += 1
            return
        if not self.in_main:
            return
        if tag in SKIP or dict(attrs).get("aria-hidden") == "true":
            self.skip += 1
            self.skip_tags.append(tag)
            return
        if self.skip:
            if self.skip_tags and tag == self.skip_tags[-1]:
                self.skip_tags.append(tag)
                self.skip += 1
            return
        if tag in INLINE and not self.pre:
            self._emit(BOUNDARY)
        elif re.fullmatch(r"h[1-6]", tag):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "pre":
            self.pre += 1
            self._emit("\n\n```\n")
        elif tag == "tr":
            self.cells = []
        elif tag in {"td", "th"}:
            self.cell = []
            self.header_row = self.header_row or tag == "th"
        elif tag == "table":
            self.rows_emitted = 0
            self._emit("\n")
        elif tag in {"dt"}:
            self._emit("\n- **")
        elif tag in {"dd"}:
            self._emit(" ")
        elif tag == "br":
            self._emit("\n")
        elif tag in BLOCK:
            self._emit("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self.in_main -= 1
            return
        if not self.in_main:
            return
        if self.skip:
            if self.skip_tags and tag == self.skip_tags[-1]:
                self.skip_tags.pop()
                self.skip -= 1
            return
        if tag in INLINE and not self.pre:
            self._emit(BOUNDARY)
        elif tag == "pre":
            self.pre -= 1
            self._emit("\n```\n")
        elif tag in {"td", "th"} and self.cell is not None and self.cells is not None:
            self.cells.append(" ".join("".join(self.cell).split()).replace("|", "\\|"))
            self.cell = None
        elif tag == "tr" and self.cells is not None:
            row = "| " + " | ".join(self.cells) + " |"
            self.out.append("\n" + row)
            if self.rows_emitted == 0:
                self.out.append("\n|" + " --- |" * len(self.cells))
            self.rows_emitted += 1
            self.cells = None
        elif tag == "dt":
            self._emit("**")
        elif re.fullmatch(r"h[1-6]", tag) or tag in BLOCK:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if not self.in_main or self.skip:
            return
        self._emit(data if self.pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        raw = "".join(self.out)
        # Inline-element boundaries become a space only where the text has none; real text is never edited.
        raw = re.sub(BOUNDARY + "+", BOUNDARY, raw)
        raw = re.sub(BOUNDARY + r"(?=[\s.,;:!?)\]]|$)", "", raw)
        raw = re.sub(r"(?:(?<=[\s(\[])|^)" + BOUNDARY, "", raw)
        raw = raw.replace(BOUNDARY, " ")
        lines = [line.rstrip() for line in raw.splitlines()]
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip() if not line.startswith("    ") else line
            if not stripped and cleaned and not cleaned[-1]:
                continue
            cleaned.append(stripped)
        return "\n".join(cleaned).strip()


def _page_path(site_root: Path, url: str, prefix: str) -> Path:
    path = urlsplit(url).path[len(prefix):]
    return site_root / (path + "index.html" if path == "" or path.endswith("/") else path)


def render(site_root: Path) -> str:
    sitemap = ElementTree.parse(site_root / "sitemap.xml").getroot()
    urls = [node.text or "" for node in sitemap.findall(".//{*}loc")]
    origin = urls[0]
    prefix = urlsplit(origin).path
    sections = [HEADER.format(sitemap=origin + "sitemap.xml")]
    for url in urls:
        parser = MainText()
        parser.feed(_page_path(site_root, url, prefix).read_text(encoding="utf-8"))
        parser.close()
        sections.append(f"\n---\n\nSource: {url}\n\n{parser.text()}\n")
    return "".join(sections)


def main() -> int:
    site_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site")
    (site_root / "llms-full.txt").write_text(render(site_root), encoding="utf-8")
    print(f"wrote {site_root / 'llms-full.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
