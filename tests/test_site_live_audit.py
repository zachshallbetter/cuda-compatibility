from __future__ import annotations

from datetime import datetime, timezone
import unittest

from scripts import site_live_audit as live

TYPES = {".html": "text/html; charset=utf-8", ".png": "image/png", ".xml": "application/xml"}


def served(overrides: dict[str, tuple[int, str, bytes]] | None = None):
    """A fake origin that serves this checkout's files, with per-URL overrides."""
    def get(url: str) -> tuple[int, str, bytes]:
        if overrides and url in overrides:
            return overrides[url]
        rel = url[len(live.ORIGIN):]
        if rel.startswith("no-such-sheet-"):
            return 404, "text/html", (live.SITE / "404.html").read_bytes()
        path = live.page_file(url, live.ORIGIN) if rel == "" or rel.endswith("/") or rel.endswith(".html") else live.SITE / rel
        want = live.FILES.get(rel) or TYPES.get(path.suffix, "application/octet-stream")
        return 200, want, path.read_bytes()
    return get


class LiveAuditTests(unittest.TestCase):
    def test_a_site_serving_the_checkout_passes(self) -> None:
        self.assertEqual(live.audit(served(), now=datetime(2026, 11, 4, tzinfo=timezone.utc)), [])

    def test_a_page_that_has_not_landed_is_reported(self) -> None:
        stale = {live.ORIGIN + "docs/cli.html": (200, "text/html", b"<html>old</html>")}
        errors = live.audit(served(stale), now=datetime(2026, 11, 4, tzinfo=timezone.utc))
        self.assertTrue(any("docs/cli.html: live bytes differ" in e for e in errors))

    def test_a_missing_share_image_is_reported(self) -> None:
        gone = {live.ORIGIN + "assets/social/cli.png": (404, "text/html", b"")}
        errors = live.audit(served(gone), now=datetime(2026, 11, 4, tzinfo=timezone.utc))
        self.assertTrue(any("assets/social/cli.png: HTTP 404" in e for e in errors))

    def test_a_wrong_content_type_is_reported(self) -> None:
        wrong = {live.ORIGIN + "feed.xml": (200, "text/xml", (live.SITE / "feed.xml").read_bytes())}
        errors = live.audit(served(wrong), now=datetime(2026, 11, 4, tzinfo=timezone.utc))
        self.assertTrue(any("/feed.xml" in e and "application/atom+xml" in e for e in errors))

    def test_an_expired_security_txt_is_reported(self) -> None:
        errors = live.audit(served(), now=datetime(2030, 1, 1, tzinfo=timezone.utc))
        self.assertTrue(any("security.txt has expired" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
