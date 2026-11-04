from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile
import unittest

from scripts.site_checks import audit_site


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = PROJECT_ROOT / "site"


class SiteDiscoveryChecksTests(unittest.TestCase):
    def copy_site(self, directory: str) -> Path:
        target = Path(directory) / "site"
        shutil.copytree(SITE_ROOT, target)
        return target

    def test_current_site_passes_discovery_checks(self) -> None:
        self.assertEqual(audit_site(SITE_ROOT), [])

    def test_rejects_missing_ga4_measurement_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            analytics = site / "analytics.js"
            analytics.write_text(
                analytics.read_text(encoding="utf-8").replace('const measurementId = "G-Q7C82JK9GM";', 'const measurementId = "";'),
                encoding="utf-8",
            )
            self.assertTrue(any("configured GA4 measurement ID" in error for error in audit_site(site)))

    def test_rejects_analytics_that_ignores_opt_outs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            analytics = site / "analytics.js"
            source = analytics.read_text(encoding="utf-8").replace(
                'if (optedOut() || !initialized || !allowedEvents.has(name)) return;',
                'if (!initialized) return;',
                1,
            )
            analytics.write_text(source, encoding="utf-8")
            self.assertTrue(any("events must respect the opt-out" in error for error in audit_site(site)))

    def test_rejects_dead_or_unlisted_analytics_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            analytics = site / "analytics.js"
            analytics.write_text(
                analytics.read_text(encoding="utf-8").replace('"hero_replay",', '"hero_replay",\n    "retired_event",', 1),
                encoding="utf-8",
            )
            script = site / "site.js"
            script.write_text(script.read_text(encoding="utf-8").replace('track("hero_replay")', 'track("hero_restart")', 1), encoding="utf-8")
            errors = audit_site(site)
            self.assertTrue(any("never emitted: retired_event" in error for error in errors))
            self.assertTrue(any("never emitted: hero_replay" in error for error in errors))
            self.assertTrue(any("not allowlisted: hero_restart" in error for error in errors))

    def test_rejects_indexable_error_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "404.html"
            page.write_text(page.read_text(encoding="utf-8").replace('content="noindex"', 'content="index"'), encoding="utf-8")
            self.assertTrue(any("404.html: the error page must declare noindex" in error for error in audit_site(site)))

    def test_rejects_stale_full_text_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "cli.html"
            page.write_text(page.read_text(encoding="utf-8").replace("List commands and example use.", "List every command.", 1), encoding="utf-8")
            self.assertTrue(any("llms-full.txt is stale" in error for error in audit_site(site)))

    def test_rejects_missing_or_wrong_size_share_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            (site / "assets" / "social" / "cli.png").write_bytes((site / "icon-192.png").read_bytes())
            errors = audit_site(site)
            self.assertTrue(any("docs/cli.html: og:image must be a local 1200x630 PNG" in error for error in errors))

    def test_rejects_page_without_touch_icon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "cli.html"
            source = page.read_text(encoding="utf-8")
            page.write_text(source.replace('<link rel="apple-touch-icon" href="../apple-touch-icon.png">\n', ""), encoding="utf-8")
            self.assertTrue(any("docs/cli.html: missing apple-touch-icon link" in error for error in audit_site(site)))

    def test_support_counts_follow_adapter_registrations(self) -> None:
        from scripts.site_checks import _adapter_registry
        registry = _adapter_registry(PROJECT_ROOT / "src" / "cuda_legacy_compat" / "torch_adapter.py")
        self.assertIsNotNone(registry)
        native, fallback = registry
        self.assertIn("mm", native)
        self.assertIn("bmm", fallback)
        self.assertNotIn("contiguous", native)

    def test_rejects_home_count_that_drifts_from_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            shutil.copytree(PROJECT_ROOT / "src", Path(directory) / "src")
            home = site / "index.html"
            source = home.read_text(encoding="utf-8")
            home.write_text(re.sub(r"(Native operation families</dt><dd[^>]*>)\d+", r"\g<1>99", source), encoding="utf-8")
            self.assertTrue(any("native operation families must read" in error for error in audit_site(site)))

    def test_rejects_missing_webmaster_verification_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            homepage = site / "index.html"
            source = homepage.read_text(encoding="utf-8")
            source = source.replace('<meta name="google-site-verification" content="2DE5E0VRt_DWCUmeZbGK35YDdealoZyJDKURNXr04ZM">', "")
            source = source.replace('<meta name="msvalidate.01" content="CDE813454E6BD389334E94C9B9C70981">', "")
            homepage.write_text(source, encoding="utf-8")
            errors = audit_site(site)
            self.assertTrue(any("google-site-verification" in error for error in errors))
            self.assertTrue(any("msvalidate.01" in error for error in errors))

    def test_rejects_stale_analytics_configuration_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            privacy = site / "docs" / "privacy.html"
            source = privacy.read_text(encoding="utf-8").replace(
                "An earlier “Reject” choice from the old prompt is still honoured.",
                "The live deployment has not yet been updated.",
                1,
            )
            privacy.write_text(source, encoding="utf-8")
            self.assertTrue(any("stale claim says configured analytics is inactive" in error for error in audit_site(site)))

    def test_rejects_stale_publication_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "discovery.html"
            source = page.read_text(encoding="utf-8").replace(
                "Point-in-time HTTP checks establish availability",
                "The public deployment has not been verified for this revision. Point-in-time HTTP checks establish availability",
                1,
            )
            page.write_text(source, encoding="utf-8")
            self.assertTrue(any("stale claim" in error for error in audit_site(site)))

    def test_rejects_missing_bing_discovery_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "discovery.html"
            source = page.read_text(encoding="utf-8").replace("Bing Webmaster Tools", "Search tools")
            page.write_text(source, encoding="utf-8")
            self.assertTrue(any("missing Bing Webmaster Tools" in error for error in audit_site(site)))

    def test_rejects_a_broken_local_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "index.html"
            page.write_text(page.read_text(encoding="utf-8").replace('href="docs/index.html"', 'href="missing.html"', 1), encoding="utf-8")
            self.assertTrue(any("broken local link" in error for error in audit_site(site)))

    def test_rejects_directory_links_that_show_file_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "index.html"
            source = page.read_text(encoding="utf-8").replace('href="docs/index.html"', 'href="docs/"', 1)
            page.write_text(source, encoding="utf-8")
            self.assertTrue(any("explicit index page" in error for error in audit_site(site)))

    def test_rejects_sitemap_that_omits_a_canonical_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            sitemap = site / "sitemap.xml"
            lines = sitemap.read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [line for line in lines if "<loc>https://cudacompat.com/docs/privacy.html</loc>" not in line]
            self.assertEqual(len(kept), len(lines) - 1)
            sitemap.write_text("".join(kept), encoding="utf-8")
            self.assertTrue(any("sitemap URLs" in error for error in audit_site(site)))

    def test_rejects_duplicate_meta_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            home = site / "index.html"
            guide = site / "docs" / "getting-started.html"
            home_source = home.read_text(encoding="utf-8")
            description = home_source.split('<meta name="description" content="', 1)[1].split('">', 1)[0]
            guide_source = guide.read_text(encoding="utf-8")
            start = guide_source.index('<meta name="description" content="')
            end = guide_source.index('">', start) + 2
            guide.write_text(guide_source[:start] + f'<meta name="description" content="{description}">' + guide_source[end:], encoding="utf-8")
            self.assertTrue(any("duplicate meta description" in error for error in audit_site(site)))

    def test_rejects_short_meta_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            homepage = site / "index.html"
            source = homepage.read_text(encoding="utf-8")
            source = re.sub(r'(<meta name="description" content=")[^"]*', r"\1Too short.", source, count=1)
            homepage.write_text(source, encoding="utf-8")
            self.assertTrue(any("meta description should be 140–180 characters" in error for error in audit_site(site)))

    def test_rejects_external_page_load_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "index.html"
            source = page.read_text(encoding="utf-8")
            page.write_text(source.replace("</head>", '<link rel="preconnect" href="https://fonts.example.net">\n</head>', 1), encoding="utf-8")
            self.assertTrue(any("page-load resource uses an external host" in error for error in audit_site(site)))

    def test_rejects_project_source_link_to_missing_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "architecture.html"
            source = page.read_text(encoding="utf-8").replace(
                "/blob/main/",
                "/blob/missing-branch/",
                1,
            )
            page.write_text(source, encoding="utf-8")
            self.assertTrue(
                any("project source link must use the published ref" in error for error in audit_site(site))
            )

    def test_rejects_article_schema_without_stable_identity_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "discovery.html"
            source = page.read_text(encoding="utf-8").replace(
                '"author":{"@id":"https://cudacompat.com/#author"}', '"author":{"@id":"https://cudacompat.com/#different-author"}', 1
            )
            page.write_text(source, encoding="utf-8")
            self.assertTrue(any("canonical #author id" in error for error in audit_site(site)))

    def test_rejects_title_outside_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "cli.html"
            source = page.read_text(encoding="utf-8").replace("<main id=\"main\">\n", "", 1).replace('<div class="wrap body"', '<main id="main" class="wrap body"', 1)
            page.write_text(source, encoding="utf-8")
            self.assertTrue(any("h1 must sit inside main" in error for error in audit_site(site)))

    def test_rejects_page_without_dark_scheme_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "cli.html"
            source = page.read_text(encoding="utf-8")
            source = source.replace('<meta name="theme-color" content="#171613" media="(prefers-color-scheme: dark)">', "", 1)
            source = source.replace('<meta name="color-scheme" content="light dark">', '<meta name="color-scheme" content="only light">', 1)
            page.write_text(source, encoding="utf-8")
            errors = audit_site(site)
            self.assertTrue(any("dark theme-color" in error for error in errors))
            self.assertTrue(any("color-scheme must be 'light dark'" in error for error in errors))

    def test_rejects_unfocusable_scroll_region(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            page = site / "docs" / "cli.html"
            page.write_text(page.read_text(encoding="utf-8").replace(' tabindex="0"', "", 1), encoding="utf-8")
            self.assertTrue(any("sideways-scrolling region" in error for error in audit_site(site)))

    def test_rejects_feed_missing_a_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            feed = site / "feed.xml"
            feed.write_text(feed.read_text(encoding="utf-8").replace("https://cudacompat.com/docs/cli.html\"", "https://cudacompat.com/docs/missing.html\"", 1), encoding="utf-8")
            self.assertTrue(any("feed.xml entries" in error for error in audit_site(site)))

    def test_rejects_expired_security_txt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = self.copy_site(directory)
            security = site / ".well-known" / "security.txt"
            security.write_text(re.sub(r"Expires: \S+", "Expires: 2020-01-01T00:00:00.000Z", security.read_text(encoding="utf-8")), encoding="utf-8")
            self.assertTrue(any("has expired" in error for error in audit_site(site)))

if __name__ == "__main__":
    unittest.main()
