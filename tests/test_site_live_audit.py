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


NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)
PINNED = live.installer_pin(live.SITE)[1]


def archive(sha: str | None = PINNED):
    return lambda url: sha


class LiveAuditTests(unittest.TestCase):
    def test_a_site_serving_the_checkout_passes(self) -> None:
        self.assertEqual(live.audit(served(), now=NOW, digest=archive()), [])

    def test_the_installer_is_audited(self) -> None:
        stale = {live.ORIGIN + "install.sh": (200, "text/plain", b"#!/bin/sh\n")}
        errors = live.audit(served(stale), now=NOW, digest=archive())
        self.assertTrue(any("/install.sh: live bytes differ" in e for e in errors), errors)

    def test_an_archive_that_does_not_match_the_pin_is_reported(self) -> None:
        url = live.installer_pin(live.SITE)[0]
        self.assertRegex(url, r"/archive/refs/tags/v\d+\.\d+\.\d+\.tar\.gz$")
        errors = live.audit(served(), now=NOW, digest=archive("0" * 64))
        self.assertTrue(any(f"install.sh pins {PINNED} for {url}, but the archive is {'0' * 64}" == e for e in errors), errors)
        errors = live.audit(served(), now=NOW, digest=archive(None))
        self.assertTrue(any("could not be downloaded" in e for e in errors), errors)

    def test_a_page_that_has_not_landed_is_reported(self) -> None:
        stale = {live.ORIGIN + "docs/cli.html": (200, "text/html", b"<html>old</html>")}
        errors = live.audit(served(stale), now=NOW, digest=archive())
        self.assertTrue(any("docs/cli.html: live bytes differ" in e for e in errors))

    def test_a_missing_share_image_is_reported(self) -> None:
        gone = {live.ORIGIN + "assets/social/cli.png": (404, "text/html", b"")}
        errors = live.audit(served(gone), now=NOW, digest=archive())
        self.assertTrue(any("assets/social/cli.png: HTTP 404" in e for e in errors))

    def test_a_wrong_content_type_is_reported(self) -> None:
        wrong = {live.ORIGIN + "feed.xml": (200, "text/xml", (live.SITE / "feed.xml").read_bytes())}
        errors = live.audit(served(wrong), now=NOW, digest=archive())
        self.assertTrue(any("/feed.xml" in e and "application/atom+xml" in e for e in errors))

    def test_an_expired_security_txt_is_reported(self) -> None:
        errors = live.audit(served(), now=datetime(2030, 1, 1, tzinfo=timezone.utc), digest=archive())
        self.assertTrue(any("security.txt has expired" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
