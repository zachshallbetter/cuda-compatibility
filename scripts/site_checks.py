"""Static release checks for the public site discovery surface."""
from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
from datetime import date
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree


# Update with the repository's default branch whenever its published ref changes.
PROJECT_SOURCE_REF = "main"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.h1_count = 0
        self.h1_text = ""
        self.ids: set[str] = set()
        self.meta: dict[str, list[str]] = {}
        self.canonicals: list[str] = []
        self.links: list[str] = []
        self.resources: list[str] = []
        self.navs: dict[str, list[str]] = {}
        self.jsonld: list[str] = []
        self._in_title = False
        self._in_h1 = False
        self._in_jsonld = False
        self._jsonld_buffer: list[str] = []
        self._nav_label: str | None = None
        self._anchor: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: value or "" for key, value in attrs}
        if data.get("id"):
            self.ids.add(data["id"])
        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self.h1_count += 1
            self._in_h1 = True
        elif tag == "meta":
            key = data.get("name") or data.get("property")
            if key:
                self.meta.setdefault(key.lower(), []).append(data.get("content", ""))
        elif tag == "link":
            rel = set(data.get("rel", "").lower().split())
            if "canonical" in rel:
                self.canonicals.append(data.get("href", ""))
            if rel & {"stylesheet", "icon", "preconnect", "dns-prefetch", "preload", "modulepreload"}:
                self.resources.append(data.get("href", ""))
            if data.get("href"):
                self.links.append(data["href"])
        elif tag == "script":
            if data.get("src"):
                self.resources.append(data["src"])
                self.links.append(data["src"])
            self._in_jsonld = data.get("type", "").lower() == "application/ld+json"
            self._jsonld_buffer = []
        elif tag == "img":
            if data.get("src"):
                self.resources.append(data["src"])
                self.links.append(data["src"])
        elif tag == "a" and data.get("href"):
            self._anchor = {"href": data["href"], "text": ""}
        elif tag == "nav" and data.get("aria-label"):
            self._nav_label = data["aria-label"]
            self.navs.setdefault(self._nav_label, [])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "a" and self._anchor is not None:
            self.links.append(self._anchor["href"])
            if self._nav_label:
                label = " ".join(self._anchor["text"].split())
                if label:
                    self.navs[self._nav_label].append(label)
            self._anchor = None
        elif tag == "nav":
            self._nav_label = None
        elif tag == "script" and self._in_jsonld:
            self.jsonld.append("".join(self._jsonld_buffer))
            self._in_jsonld = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._in_h1:
            self.h1_text += data
        if self._anchor is not None:
            self._anchor["text"] += data
        if self._in_jsonld:
            self._jsonld_buffer.append(data)


def _parse(path: Path) -> tuple[PageParser, str]:
    parser = PageParser()
    source = path.read_text(encoding="utf-8")
    parser.feed(source)
    parser.close()
    return parser, source


def _expected_url(relative: Path, origin: str, prefix: str) -> str:
    local = relative.as_posix()
    if local == "index.html":
        suffix = ""
    elif local.endswith("/index.html"):
        suffix = local[: -len("index.html")]
    else:
        suffix = local
    return f"{origin}{prefix}{suffix}"


def _adapter_registry(source: Path) -> tuple[set[str], set[str]] | None:
    """Return (native operation families, CPU fallback routes) registered by the PyTorch adapter."""
    try:
        code = source.read_text(encoding="utf-8")
    except OSError:
        return None
    native: set[str] = set()
    fallback: set[str] = set()
    for name, handler in re.findall(r'_library\.impl\(\s*"([^"]+)",\s*(_cpu_fallback\(|_[a-z0-9_]+)', code):
        if handler == "_cpu_fallback(":
            fallback.add(name)
            continue
        body = re.search(rf"\ndef {handler}\(.*?(?=\ndef |\Z)", code, re.S)
        text = body.group(0) if body else ""
        if 'last_path = "llvm-sm52' in text:
            native.add(name.split(".")[0].rstrip("_"))
        elif "cpu-fallback" in text:
            fallback.add(name)
    return native, fallback


def _audit_support_counts(site_root: Path) -> list[str]:
    """Published support counts must agree with the adapter's registrations; no count is hardcoded here."""
    registry = _adapter_registry(site_root.parent / "src" / "cuda_legacy_compat" / "torch_adapter.py")
    if registry is None:
        return []
    families, fallbacks = len(registry[0]), len(registry[1])
    errors: list[str] = []
    try:
        home = (site_root / "index.html").read_text(encoding="utf-8")
        matrix = (site_root / "docs" / "compatibility.html").read_text(encoding="utf-8")
    except OSError as exc:
        return [f"support count sources are unreadable: {exc}"]
    stat = re.search(r"<dt[^>]*>Native operation families</dt><dd[^>]*>(\d+)</dd>", home)
    if not stat or int(stat.group(1)) != families:
        errors.append(f"index.html: native operation families must read {families} (adapter registrations)")
    cases = re.search(r"These (\d+) registered cases cover (\d+) operation families", matrix)
    table = re.search(r'<table aria-label="Native GPU implementations">(.*?)</table>', matrix, re.S)
    rows = len(re.findall(r"<tr>", table.group(1).split("</thead>")[-1])) if table else -1
    if not cases or int(cases.group(2)) != families or int(cases.group(1)) != rows:
        errors.append(f"docs/compatibility.html: native summary must state {rows} cases across {families} families")
    routes = re.search(r"These (\d+) registered fallback routes", matrix)
    if not routes or int(routes.group(1)) != fallbacks:
        errors.append(f"docs/compatibility.html: fallback summary must state {fallbacks} registered routes")
    return errors


def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _load_llms_full_builder():
    spec = importlib.util.spec_from_file_location("build_llms_full", Path(__file__).with_name("build_llms_full.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit_event_allowlist(analytics_source: str, site_script: Path) -> list[str]:
    """Allowlisted analytics events and events the site can emit must match exactly."""
    allowlist = re.search(r"allowedEvents = new Set\(\[(.*?)\]\)", analytics_source, re.S)
    classifier = re.search(r"const classifyLink = .*?\n  \};", analytics_source, re.S)
    if not allowlist or not classifier:
        return ["analytics.js: expected an allowedEvents set and a classifyLink function"]
    allowed = set(re.findall(r'"([a-z_]+)"', allowlist.group(1)))
    emitted = set(re.findall(r'return "([a-z_]+)";', classifier.group(0)))
    try:
        emitted |= set(re.findall(r'track\("([a-z_]+)"', site_script.read_text(encoding="utf-8")))
    except OSError as exc:
        return [f"site.js is missing or unreadable: {exc}"]
    errors = [f"analytics.js: allowlisted event is never emitted: {name}" for name in sorted(allowed - emitted)]
    errors += [f"analytics event is emitted but not allowlisted: {name}" for name in sorted(emitted - allowed)]
    return errors


def audit_site(site_root: Path) -> list[str]:
    site_root = site_root.resolve()
    errors: list[str] = []
    not_found = site_root / "404.html"
    pages = sorted(page for page in site_root.rglob("*.html") if page != not_found)
    home = site_root / "index.html"
    if not home.is_file():
        return ["site/index.html is missing"]
    if not pages:
        return ["site has no HTML pages"]

    parsed: dict[Path, tuple[PageParser, str]] = {page: _parse(page) for page in pages}
    analytics_path = site_root / "analytics.js"
    try:
        analytics_source = analytics_path.read_text(encoding="utf-8")
    except OSError as exc:
        analytics_source = ""
        errors.append(f"analytics.js is missing or unreadable: {exc}")
    if not re.search(r'const measurementId = "G-[A-Z0-9]+";', analytics_source):
        errors.append("analytics.js: expected a configured GA4 measurement ID")
    if 'if (!optedOut()) loadAnalytics();' not in analytics_source:
        errors.append("analytics.js: the tag must not load for a browser that opted out")
    if 'navigator.globalPrivacyControl === true || navigator.doNotTrack === "1"' not in analytics_source:
        errors.append("analytics.js: Global Privacy Control and Do Not Track must count as an opt-out")
    if 'if (optedOut() || !initialized || !allowedEvents.has(name)) return;' not in analytics_source:
        errors.append("analytics.js: events must respect the opt-out and use an allowlisted name")
    if "analytics-banner" in analytics_source:
        errors.append("analytics.js: the site has no analytics prompt; remove the banner")

    errors.extend(_audit_event_allowlist(analytics_source, site_root / "site.js"))
    errors.extend(_audit_support_counts(site_root))

    if not_found.is_file():
        not_found_meta = _parse(not_found)[0].meta
        if not any("noindex" in value.lower() for value in not_found_meta.get("robots", [])):
            errors.append("404.html: the error page must declare noindex")

    homepage_meta = parsed[home][0].meta
    for name in ("google-site-verification", "msvalidate.01"):
        values = homepage_meta.get(name, [])
        if len(values) != 1 or not values[0].strip():
            errors.append(f"site/index.html: expected one non-empty {name} verification tag")

    privacy_page = site_root / "docs" / "privacy.html"
    if privacy_page in parsed:
        privacy_source = parsed[privacy_page][1].lower()
        if "live deployment has not yet been updated" in privacy_source or "no measurement id" in privacy_source:
            errors.append("docs/privacy.html: stale claim says configured analytics is inactive")

    discovery_page = site_root / "docs" / "discovery.html"
    if discovery_page in parsed:
        discovery_source = parsed[discovery_page][1].lower()
        if "has not been verified for this revision" in discovery_source:
            errors.append("docs/discovery.html: stale claim says the public deployment is unverified")
        if "bing webmaster tools" not in discovery_source:
            errors.append("docs/discovery.html: missing Bing Webmaster Tools setup guidance")
        home_canonicals = parsed[home][0].canonicals
        if not home_canonicals or home_canonicals[0].lower() not in discovery_source:
            errors.append("docs/discovery.html: missing the exact hosted site URL-prefix property")
        if "site has no ga4 measurement id" in discovery_source or "ga4 remains inactive" in discovery_source:
            errors.append("docs/discovery.html: stale claim says GA4 is unconfigured")
    home_parser, _ = parsed[home]
    if len(home_parser.canonicals) != 1:
        return ["homepage must have exactly one canonical URL"]
    home_url = home_parser.canonicals[0]
    home_parts = urlsplit(home_url)
    if home_parts.scheme != "https" or not home_parts.netloc or not home_parts.path.endswith("/"):
        errors.append("homepage canonical must be an HTTPS URL ending in /")
        return errors
    origin = f"{home_parts.scheme}://{home_parts.netloc}"
    prefix = home_parts.path
    canonical_urls: set[str] = set()
    page_ids = {path: parser.ids for path, (parser, _) in parsed.items()}
    titles: set[str] = set()
    descriptions: set[str] = set()
    primary_nav: list[str] | None = None
    docs_nav: list[str] | None = None

    for page, (parser, source) in parsed.items():
        rel = page.relative_to(site_root)
        label = rel.as_posix()
        expected = _expected_url(rel, origin, prefix)
        if len(parser.canonicals) != 1:
            errors.append(f"{label}: expected exactly one canonical URL")
        elif parser.canonicals[0] != expected:
            errors.append(f"{label}: canonical {parser.canonicals[0]!r} does not match {expected!r}")
        canonical = parser.canonicals[0] if len(parser.canonicals) == 1 else ""
        if canonical:
            canonical_urls.add(canonical)
        title = " ".join(parser.title.split())
        if not title:
            errors.append(f"{label}: missing title")
        elif title in titles:
            errors.append(f"{label}: duplicate title {title!r}")
        titles.add(title)
        if parser.h1_count != 1:
            errors.append(f"{label}: expected one h1, found {parser.h1_count}")
        if len(parser.meta.get("description", [])) != 1 or not parser.meta["description"][0].strip():
            errors.append(f"{label}: expected one non-empty meta description")
        else:
            description = parser.meta["description"][0].strip()
            if not 140 <= len(description) <= 180:
                errors.append(f"{label}: meta description should be 140–180 characters")
            if description in descriptions:
                errors.append(f"{label}: duplicate meta description")
            else:
                descriptions.add(description)
        if not " ".join(parser.h1_text.split()):
            errors.append(f"{label}: h1 must contain visible text")
        for key in ("og:title", "og:description", "og:url", "og:image", "twitter:card", "twitter:title", "twitter:description", "twitter:image"):
            if len(parser.meta.get(key, [])) != 1 or not parser.meta[key][0].strip():
                errors.append(f"{label}: expected one non-empty {key} value")
        if canonical and parser.meta.get("og:url") != [canonical]:
            errors.append(f"{label}: og:url does not match canonical")
        robots = parser.meta.get("robots", []) + parser.meta.get("googlebot", [])
        if not parser.meta.get("robots"):
            errors.append(f"{label}: missing robots policy")
        if any("noindex" in value.lower() for value in robots):
            errors.append(f"{label}: indexable page declares noindex")
        if not re.search(r'<main\b', source, re.I) or not re.search(r'<a\b[^>]*class="skip-link"', source, re.I):
            errors.append(f"{label}: missing semantic main or skip link")
        main_at, h1_at = source.find("<main"), source.find("<h1")
        if main_at < 0 or h1_at < main_at:
            errors.append(f"{label}: the h1 must sit inside main so skip links and reader views start at the title")
        if "TechArticle" in source and not re.search(r'<time datetime="\d{4}-\d\d-\d\d">', source):
            errors.append(f"{label}: article pages must show their updated date in a <time> element")
        for wrapper in re.findall(r'<div\b[^>]*overflow-x:auto[^>]*>', source):
            if 'tabindex="0"' not in wrapper or "aria-label=" not in wrapper:
                errors.append(f"{label}: a sideways-scrolling region must be focusable and labelled")
        if 'name="theme-color" content="#F2F2EE"' not in source:
            errors.append(f"{label}: theme-color must match the paper colour #F2F2EE")
        if not any("privacy.html" in href for href in parser.links):
            errors.append(f"{label}: missing privacy and measurement link")
        if not parser.jsonld:
            errors.append(f"{label}: missing JSON-LD")
        share_image = (parser.meta.get("og:image") or [""])[0]
        if share_image.startswith(origin + prefix):
            image_path = site_root / share_image[len(origin + prefix):]
            if _png_size(image_path) != (1200, 630):
                errors.append(f"{label}: og:image must be a local 1200x630 PNG: {share_image}")
        elif share_image:
            errors.append(f"{label}: og:image must be served from the canonical origin")
        if parser.meta.get("twitter:image") != parser.meta.get("og:image"):
            errors.append(f"{label}: twitter:image does not match og:image")
        rels = re.findall(r'<link\b[^>]*\brel="([^"]+)"', source, re.I)
        for required in ("icon", "apple-touch-icon", "manifest"):
            if required not in rels:
                errors.append(f"{label}: missing {required} link")
        for raw in parser.jsonld:
            try:
                structured = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{label}: invalid JSON-LD: {exc}")
                continue
            if canonical and canonical not in json.dumps(structured, sort_keys=True):
                errors.append(f"{label}: JSON-LD does not identify its canonical page")
            nodes = structured.get("@graph", [structured]) if isinstance(structured, dict) else []
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if node.get("@type") in {"Article", "TechArticle"}:
                    if node.get("@id") != f"{canonical}#article":
                        errors.append(f"{label}: article JSON-LD must have a stable canonical #article id")
                    if (node.get("author") or {}).get("@id") != f"{origin}{prefix}#author":
                        errors.append(f"{label}: article author must use the canonical #author id")
                    if (node.get("isPartOf") or {}).get("@id") != f"{origin}{prefix}#website":
                        errors.append(f"{label}: article must refer to the canonical #website id")
                if rel.as_posix() == "index.html" and node.get("@type") in {"WebSite", "Person", "SoftwareApplication"}:
                    expected_id = {
                        "WebSite": f"{origin}{prefix}#website",
                        "Person": f"{origin}{prefix}#author",
                        "SoftwareApplication": f"{origin}{prefix}#software",
                    }[node["@type"]]
                    if node.get("@id") != expected_id:
                        errors.append(f"homepage: {node['@type']} must use stable id {expected_id}")
        for key, nav in parser.navs.items():
            if key == "Main navigation":
                if primary_nav is None:
                    primary_nav = nav
                elif nav != primary_nav:
                    errors.append(f"{label}: primary navigation differs from other pages")
            elif key == "Documentation pages":
                if docs_nav is None:
                    docs_nav = nav
                elif nav != docs_nav:
                    errors.append(f"{label}: documentation navigation differs across pages")

        for resource in parser.resources:
            parts = urlsplit(resource)
            if parts.scheme or parts.netloc:
                errors.append(f"{label}: page-load resource uses an external host: {resource}")
        for href in parser.links:
            parts = urlsplit(href)
            if parts.netloc.lower() == "github.com" and parts.path.startswith(
                "/zachshallbetter/cuda-compatibility/blob/"
            ):
                source_path = parts.path.split("/blob/", 1)[1]
                uses_published_ref = source_path.startswith(PROJECT_SOURCE_REF + "/")
                uses_commit = bool(re.match(r"^[0-9a-f]{40}/", source_path))
                if not uses_published_ref and not uses_commit:
                    errors.append(
                        f"{label}: project source link must use the published ref or an immutable commit: {href}"
                    )
            if parts.scheme or parts.netloc or href.startswith("//"):
                continue
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            if target.is_dir():
                if parts.path.endswith("/"):
                    errors.append(f"{label}: local link targets a directory instead of an explicit index page: {href}")
                target = target / "index.html"
            try:
                target.relative_to(site_root)
            except ValueError:
                errors.append(f"{label}: local link escapes site root: {href}")
                continue
            if not target.is_file():
                errors.append(f"{label}: broken local link: {href}")
            elif parts.fragment and target.suffix.lower() == ".html":
                target_ids = page_ids.get(target)
                if target_ids is None:
                    target_ids = _parse(target)[0].ids
                if unquote(parts.fragment) not in target_ids:
                    errors.append(f"{label}: missing fragment target: {href}")

    sitemap = site_root / "sitemap.xml"
    try:
        root = ElementTree.parse(sitemap).getroot()
        locations = [node.text or "" for node in root.findall("{*}url/{*}loc")]
        sitemap_urls = set(locations)
        if len(locations) != len(sitemap_urls):
            errors.append("sitemap.xml contains duplicate URLs")
        if sitemap_urls != canonical_urls:
            errors.append("sitemap URLs do not exactly match canonical HTML pages")
    except (OSError, ElementTree.ParseError) as exc:
        errors.append(f"sitemap.xml is missing or invalid: {exc}")

    robots = site_root / "robots.txt"
    try:
        robots_text = robots.read_text(encoding="utf-8")
        if f"Sitemap: {origin}{prefix}sitemap.xml" not in robots_text:
            errors.append("robots.txt does not point to the canonical sitemap")
        if not re.search(r"(?im)^User-agent:\s*\*\s*\nAllow:\s*/\s*$", robots_text):
            errors.append("robots.txt must explicitly allow the general crawler group")
    except OSError as exc:
        errors.append(f"robots.txt is missing: {exc}")

    llms = site_root / "llms.txt"
    try:
        llms_urls = set(re.findall(r"https://[^\s)]+", llms.read_text(encoding="utf-8")))
        site_urls = {
            url for url in llms_urls
            if url.startswith(origin + prefix) and (url.endswith("/") or url.endswith(".html"))
        }
        if f"{origin}{prefix}llms-full.txt" not in llms_urls:
            errors.append("llms.txt does not point to llms-full.txt")
        if site_urls != canonical_urls:
            errors.append("llms.txt URLs do not exactly match canonical HTML pages")
    except OSError as exc:
        errors.append(f"llms.txt is missing: {exc}")

    manifest = site_root / "manifest.webmanifest"
    try:
        icons = json.loads(manifest.read_text(encoding="utf-8")).get("icons", [])
        if not icons:
            errors.append("manifest.webmanifest declares no icons")
        for icon in icons:
            size = _png_size(site_root / str(icon.get("src", "")).lstrip("/"))
            declared = str(icon.get("sizes", ""))
            if size is None or f"{size[0]}x{size[1]}" != declared:
                errors.append(f"manifest.webmanifest: icon {icon.get('src')} is missing or not {declared}")
    except (OSError, ValueError) as exc:
        errors.append(f"manifest.webmanifest is missing or invalid: {exc}")

    full_text = site_root / "llms-full.txt"
    try:
        builder = _load_llms_full_builder()
        if full_text.read_text(encoding="utf-8") != builder.render(site_root):
            errors.append("llms-full.txt is stale; run python3 scripts/build_llms_full.py")
    except OSError as exc:
        errors.append(f"llms-full.txt is missing: {exc}")

    try:
        feed = ElementTree.parse(site_root / "feed.xml").getroot()
        feed_urls = {node.get("href") for node in feed.findall("{*}entry/{*}link")}
        if feed_urls != canonical_urls:
            errors.append("feed.xml entries do not exactly match canonical HTML pages")
    except (OSError, ElementTree.ParseError) as exc:
        errors.append(f"feed.xml is missing or invalid: {exc}")
    if not (site_root / "humans.txt").is_file():
        errors.append("humans.txt is missing")
    try:
        security = (site_root / ".well-known" / "security.txt").read_text(encoding="utf-8")
        expires = re.search(r"(?m)^Expires: (\d{4}-\d\d-\d\d)", security)
        if not re.search(r"(?m)^Contact: https://", security) or not expires:
            errors.append(".well-known/security.txt needs Contact and Expires fields")
        elif expires[1] <= date.today().isoformat():
            errors.append(".well-known/security.txt has expired; move Expires forward")
    except OSError as exc:
        errors.append(f".well-known/security.txt is missing: {exc}")

    css = site_root / "styles.css"
    if css.is_file() and re.search(r"(?:@import\s+|url\()\s*['\"]?https?://", css.read_text(encoding="utf-8"), re.I):
        errors.append("styles.css declares an external page-load resource")
    return errors


def main() -> int:
    errors = audit_site(Path("site"))
    if errors:
        print("INVALID SITE DISCOVERY SURFACE\n" + "\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("OK site discovery: canonical pages, metadata, navigation, links, sitemap, robots, and agent index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
