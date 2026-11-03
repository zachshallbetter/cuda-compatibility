from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import indexnow


class IndexNowTests(unittest.TestCase):
    def test_canonical_urls_map_to_committed_files(self) -> None:
        origin = "https://cudacompat.com/"
        self.assertEqual(indexnow.page_file(origin, origin), indexnow.SITE / "index.html")
        self.assertEqual(indexnow.page_file(origin + "docs/", origin), indexnow.SITE / "docs" / "index.html")
        self.assertEqual(indexnow.page_file(origin + "docs/cli.html", origin), indexnow.SITE / "docs" / "cli.html")

    def test_every_sitemap_url_has_a_committed_page(self) -> None:
        urls = indexnow.sitemap_urls()
        for url in urls:
            self.assertTrue(indexnow.page_file(url, urls[0]).is_file(), url)

    def test_only_changed_pages_are_sent(self) -> None:
        urls = indexnow.sitemap_urls()
        changed = indexnow.changed_urls(["site/docs/cli.html", "site/styles.css", "README.md"], urls)
        self.assertEqual(changed, [urls[0] + "docs/cli.html"])
        self.assertEqual(indexnow.changed_urls(["site/home.css", "site/site.js"], urls), [])

    def test_payload_uses_the_root_key_file(self) -> None:
        body = indexnow.payload(["https://cudacompat.com/"])
        self.assertEqual(body["host"], "cudacompat.com")
        self.assertEqual(body["keyLocation"], f"https://cudacompat.com/{body['key']}.txt")

    def test_nothing_is_sent_when_no_page_changed(self) -> None:
        with mock.patch.object(indexnow, "git_changed", return_value=["site/styles.css"]), \
                mock.patch.object(indexnow, "urlopen") as opened:
            self.assertEqual(indexnow.main(["--since", "HEAD~1", "--wait-live", "0"]), 0)
        opened.assert_not_called()

    def test_waits_for_the_deploy_to_serve_committed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "page.html"
            page.write_bytes(b"new")
            served = iter([b"old", b"new"])
            with mock.patch.object(indexnow, "fetch", side_effect=lambda url: next(served)), \
                    mock.patch.object(indexnow.time, "sleep"):
                self.assertTrue(indexnow.wait_live({"https://cudacompat.com/": page}, 60, interval=0))
            with mock.patch.object(indexnow, "fetch", return_value=b"old"):
                self.assertFalse(indexnow.wait_live({"https://cudacompat.com/": page}, 0))

    def test_a_deploy_that_never_lands_sends_nothing(self) -> None:
        with mock.patch.object(indexnow, "git_changed", return_value=["site/index.html"]), \
                mock.patch.object(indexnow, "wait_live", return_value=False), \
                mock.patch.object(indexnow, "urlopen") as opened:
            self.assertEqual(indexnow.main(["--since", "HEAD~1", "--wait-live", "0"]), 1)
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
