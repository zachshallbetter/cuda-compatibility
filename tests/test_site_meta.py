from __future__ import annotations

import json
import unittest
from unittest import mock

from scripts import build_site_meta as meta


def same(target: str) -> str:
    return target


class MarkdownTests(unittest.TestCase):
    def test_headings_tables_code_and_links(self) -> None:
        h1, body, toc = meta.markdown(
            "# Title\n\n## First part\n\nText with `code`, **bold** and [a link](x.md).\n\n"
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n```sh\necho <hi>\n```\n\n- one\n- two\n", same)
        self.assertEqual(h1, "Title")
        self.assertEqual(toc, [("first-part", "First part")])
        self.assertIn('<h2 id="first-part">First part</h2>', body)
        self.assertIn("<code>code</code>", body)
        self.assertIn("<strong>bold</strong>", body)
        self.assertIn('<a href="x.md">a link</a>', body)
        self.assertIn('role="region" aria-label="First part (scrolls sideways)" tabindex="0"', body)
        self.assertIn('<th scope="col">A</th>', body)
        self.assertIn("echo &lt;hi&gt;", body)
        self.assertIn('<ul class="list"><li>one</li><li>two</li></ul>', body)

    def test_bare_urls_become_links(self) -> None:
        _, body, _ = meta.markdown("- https://docs.nvidia.com/cuda/x.html\n", same)
        self.assertIn('<a href="https://docs.nvidia.com/cuda/x.html">https://docs.nvidia.com/cuda/x.html</a>', body)

    def test_a_code_span_can_be_link_text(self) -> None:
        _, body, _ = meta.markdown("Run [`run_native_baseline.sh`](../../x/run.sh), then `make` and [`a<b`](y).\n", same)
        self.assertIn('<a href="../../x/run.sh"><code>run_native_baseline.sh</code></a>', body)
        self.assertIn("<code>make</code>", body)
        self.assertIn('<a href="y"><code>a&lt;b</code></a>', body)
        self.assertNotIn("](", body)

    def test_link_text_can_mix_code_and_words(self) -> None:
        _, body, _ = meta.markdown("See [`a` and `b`](y.md), [plain](z.md) then `c`, and `[not](a link)`.\n", same)
        self.assertIn('<a href="y.md"><code>a</code> and <code>b</code></a>', body)
        self.assertIn('<a href="z.md">plain</a> then <code>c</code>', body)
        self.assertIn("<code>[not](a link)</code>", body)

    def test_no_generated_page_shows_a_literal_markdown_link(self) -> None:
        import re
        from pathlib import Path
        for page in sorted((Path(meta.__file__).resolve().parents[1] / "site").rglob("*.html")):
            text = re.sub(r"(?s)<(pre|script|style)\b.*?</\1>|<[^>]+>", " ", page.read_text(encoding="utf-8"))
            self.assertNotRegex(text, r"\]\((?:\.\./|https?:|[\w-]+/)", page.name)


class RegistryTests(unittest.TestCase):
    def test_current_registry_and_pages_are_clean(self) -> None:
        self.assertEqual(meta.problems(), [])

    def test_every_page_has_one_excerpt_used_everywhere(self) -> None:
        for page in meta.load()["pages"]:
            text = (meta.SITE / page["file"]).read_text(encoding="utf-8")
            excerpt = meta.escape(page["excerpt"])
            for tag in ('name="description"', 'property="og:description"', 'name="twitter:description"'):
                self.assertIn(f'<meta {tag} content="{excerpt}">', text, f'{page["file"]} {tag}')

    def test_rejects_an_excerpt_too_long_for_search(self) -> None:
        data = json.loads(meta.REGISTRY.read_text(encoding="utf-8"))
        data["pages"][0]["excerpt"] = "x" * 170
        with mock.patch.object(meta, "load", return_value=self._loaded(data)), mock.patch.object(meta, "build", return_value={}):
            self.assertTrue(any("write 140–158" in e for e in meta.problems()))

    def test_rejects_a_duplicate_excerpt(self) -> None:
        data = json.loads(meta.REGISTRY.read_text(encoding="utf-8"))
        data["pages"][1]["excerpt"] = data["pages"][0]["excerpt"]
        with mock.patch.object(meta, "load", return_value=self._loaded(data)), mock.patch.object(meta, "build", return_value={}):
            self.assertTrue(any("duplicates another page" in e for e in meta.problems()))

    @staticmethod
    def _loaded(data: dict) -> dict:
        for page in data["pages"]:
            page["url"] = meta.ORIGIN + page["path"]
            page["image"] = meta.ORIGIN + meta.card_path(page)
            page["alt"] = page["share_title"]
        return data


if __name__ == "__main__":
    unittest.main()
