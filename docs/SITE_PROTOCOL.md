# Site protocol

How cudacompat.com is changed, checked, shipped and kept. It applies to every change under `site/` and to the scripts that build or check it.

[`DPP_IMPLEMENTATION.md`](DPP_IMPLEMENTATION.md) is the register of what is true about each surface. This file is the procedure that keeps it true. Work items still follow [`AGENTS.md`](../AGENTS.md), and pushes, merges and deploys still need ACP `ALLOW` (§17 there).

## 1. The goal

Any single page that is opened, shared, printed, read aloud, crawled or fetched by an agent must say, on its own:
- what it is
- who wrote it
- when it changed
- where it sits in the site
- what to do next

It must say this without the rest of the site, without JavaScript, and at any width from 320 px.

## 2. Titles and excerpts

`content/pages.json` holds, for every canonical page:
- the search title, at most 60 characters
- the share title, at most 70 characters
- an eyebrow and a card chip
- **one excerpt**

`scripts/build_site_meta.py` writes the excerpt into the meta description, the Open Graph and X descriptions, the schema.org descriptions (and `abstract` on papers), the feed summary, `llms.txt` and the share card. Never write a description into a page by hand. `make check` fails when a page drifts from the registry.

Write the excerpt for someone deciding in two seconds whether to open the page:
- 140–158 characters, so search results don't cut it off and LinkedIn's 100-character minimum is met
- lead with what the reader gets or learns, in plain words
- one specific fact, number or name where one exists
- no claim the page doesn't support: limits travel with claims
- unique across the site

Research papers live in `docs/` as Markdown, and `build_site_meta.py` renders each one to `site/research/<slug>.html`. Each paper page gets the full head, `citation_*` tags for scholarly indexes, a `ScholarlyArticle` graph whose `isBasedOn` is the source file, previous and next links, a citation line and its own card. To add a paper, add a registry entry with `kind: paper`, `source`, `published` and `source_modified`, then run `make site` and `make cards`. The research sheet lists every paper with its excerpt.

## 3. Page contract

Every canonical page meets all of these. **Check** is what enforces it: `make check` (`scripts/site_checks.py` and `scripts/build_site_meta.py --check`), a test, or a manual step in §6.

### Head

| Requirement | Check |
|---|---|
| `lang="en"`, charset, viewport, one `<title>`, one meta description of 140–180 characters that no other page shares | `make check` |
| `<link rel="canonical">` to the page's own URL. `og:url` is the same URL | `make check`; live audit |
| `robots` policy, and no `noindex` on an indexable page | `make check` |
| `color-scheme` `light dark`; two `theme-color` tags, `#F2F2EE` with `media="(prefers-color-scheme: light)"` and `#171613` with `media="(prefers-color-scheme: dark)"` | `make check` (theme-color, color-scheme) |
| Favicon ICO and SVG, apple-touch icon, mask icon, manifest | `make check` |
| `rel=alternate` (the feed), `rel=author` (`humans.txt`), `rel=license`, `rel=prev` and `rel=next` along the sheet order | manual |
| Open Graph: title, description of at least 100 characters (LinkedIn's minimum), url, type, site name, locale, and image with type, size and alt. Twitter card tags. The Twitter image equals the OG image | `make check`, except the 100-character minimum, which LinkedIn reports (§6.6) |
| On article pages: `article:published_time` (the file's first commit) and `article:modified_time`, both full ISO timestamps with an offset, plus `article:author` and `article:section` | manual |
| One JSON-LD `@graph` holding the full `WebSite`, `Person` (`#author`), `Organization` (`#org`), `WebPage`, `ImageObject`, `BreadcrumbList` and a `TechArticle` (`#article`). The home page uses `SoftwareApplication` (`#software`, with an `Offer`) and `SoftwareSourceCode` (`#source`) instead of the article. Stable `@id`s only | `make check` (ids, canonical) |

### Body

| Requirement | Check |
|---|---|
| Skip link, then the banner and main navigation, then `main`. On sheets the order inside `main` is `article` > `header.band` (breadcrumb, the one `h1`, summary, byline), then the documentation navigation, the content, and the previous/next links. The footer title block comes last | `make check` (h1 inside main) |
| Byline: author link, sheet number, and `Updated` with `<time datetime>` | `make check` |
| Visible breadcrumb equal to the `BreadcrumbList` | `make check` |
| Headings never skip a level. Every table has a name and `scope` on its header cells. A wide table sits in a labelled `role=region` with `tabindex="0"` | `make check` (regions); §6.5 |
| Every link and button has a name. No link text points to two different places. Focus is visible on every control | §6.5 |
| WCAG 2.2 AA contrast, and targets of at least 24 px | §6.5 |
| Reads fully without JavaScript. Script only enhances: the hero animation and the copy buttons, which stay hidden until script shows them. The hero falls back to `assets/gm200-wireframe.png` through a `<noscript>` style | §6.4 |
| No horizontal page scroll at 320, 375 or 768 px | §6.4 |
| Print: ink on white, navigation hidden, link targets written out, rows unbroken | §6.4 |
| No third-party page-load resource apart from analytics, which never loads for visitors who opt out (Global Privacy Control, Do Not Track, or the privacy-page switch). There is no consent prompt | `make check` |

### Look

- Ink on paper only. Every colour is a token on `:root` in `styles.css`; pages and per-page styles use `var(--…)`, never a literal, except black text on a yellow fill.
- Light: paper `#F2F2EE` everywhere (body, header, bands, footer), raised surfaces (cards, code blocks, notes, the pager) `#E9E9E3`, inline code `#E2E2DA`, text `#111`, lines and link ink black, labels `#4A4A46`, muted text `#3A3A36`, accent, hover and focus blue `#1F3DFF`. **No white-on-black section.**
- Dark (`prefers-color-scheme: dark`, screen only): paper `#171613`, surfaces `#211F1B`, inline code `#2C2A25`, text `#ECEAE3`, ink `#F6F4ED`, lines `#D9D6CD`, labels `#ABA89F`, muted text `#C9C6BD`, accent and focus `#8C9CFF`, contradiction red `#FF7A85`. The caution fill (`.y`, `.note.y`, the reference row) turns dark amber `#2E2808` with a yellow line. The hero wireframe reads its strokes from `--wire-ink`, `--wire-paper` and `--wire-accent`; the no-script PNG is inverted.
- Print is always light: the dark tokens apply to `screen` only.
- Yellow `#FFD400` is a fill under black text, always bounded by an ink line. It is never text colour.
- Scrollbars, text selection and tap highlight are themed site-wide in `styles.css`, in both schemes.
- No colour carries meaning alone. Every state has a mark shape and a text label.

### Share card

A 1200×630 PNG per page, ink on paper. It shows the page's eyebrow, share title, excerpt, chip and URL, all taken from the registry; the title shrinks until the text clears the footer rule. `scripts/build_social_cards.py` renders every card, so **never edit a card by hand**: change the page, then run `make cards`. Rendering is deterministic, so cards for unchanged pages stay byte-identical.

## 4. Site files

| File | Rule | Check |
|---|---|---|
| `robots.txt` | Explicitly allows `*`, keeps the per-crawler choices, points to the sitemap | `make check` |
| `sitemap.xml` | Exactly the canonical pages, a true `lastmod`, one `image:image` per page. Readers take only `url/loc`, never image locations | `make check` |
| `feed.xml` | Atom, one entry per canonical page, a true `updated` | `make check` |
| `llms.txt` | Every canonical page and the machine-readable files | `make check` |
| `llms-full.txt` | Generated by `scripts/build_llms_full.py`; never edited by hand | `make check` (freshness) |
| `humans.txt` | People, credits, standards, last update | `make check` (present) |
| `.well-known/security.txt` | RFC 9116, with Contact, Expires, Canonical and Policy (`SECURITY.md`). **Renew `Expires` before it passes** (now 2027-11-03); `make check` fails on that date | `make check` |
| `manifest.webmanifest` | id, name, lang, paper colours, PNG icons with `purpose` | `make check` (icons) |
| `404.html` | `noindex`, outside the sitemap, served for unknown paths | `make check`; live audit |
| IndexNow key (`03dd…ee.txt`) | One key at the root | `scripts/indexnow.py` |
| `vercel.json` | Security headers, plain-text and Atom content types, font caching | live audit |
| `SECURITY.md`, `CITATION.cff` (repository root) | Kept in step with the release version | manual |

## 5. Change procedure

1. **Isolate.** Work in `~/Projects/_worktrees/cuda-compatibility/<topic>` on a new branch from `origin/main`. Check `git log origin/main` first: other sessions merge here quickly.
2. **Edit** the pages and scripts. Keep every page in step: when a rule applies to all pages, change all of them in the same commit.
3. **Regenerate and gate.** Run `make cards` if a title, summary, sheet number or card design changed. Then run `make site`, which writes the registry into the pages, renders the papers, regenerates `llms-full.txt` and the context lock and then runs `make check` and `make test`. Both must pass. On a machine with Google Chrome, also run `make phone-width`: every page must fit a 390 px screen in light and dark, with no literal Markdown link. Update a test only when its fixture no longer matches the page's real shape; never weaken a check to pass.
4. **Render** as in §6.4. Look before you claim.
5. **Commit** as the project owner, with no AI attribution anywhere. Open a PR whose description lists what changed and what was not verified.
6. **Merge** only when the PR is `CLEAN` with all three checks green (Verify site and project, Vercel, Vercel Preview Comments). Use `gh pr merge N --squash --delete-branch --match-head-commit <head>`. **A merge to `main` deploys production**, usually within about 40 seconds.
7. **Verify live.** Run `make live-audit`. It must print `OK live site matches the checkout`. If a page hasn't landed, wait and run it again; don't claim a deploy that hasn't been seen.
8. **Notify.** `.github/workflows/indexnow.yml` sends the changed pages to IndexNow on its own once they are live. Confirm the run's `IndexNow responded 200` line. Nothing is sent when only CSS or JavaScript changed.
9. **Preview** (§6.6) when a title, description or share card changed.
10. **Record** (§7), then remove the worktree and branch.

### Adding, removing or renaming a page

Update all of these in the same change:
- the page itself, built from an existing sheet
- `sitemap.xml`, `feed.xml` and `llms.txt`
- the documentation navigation on every sheet, and the main navigation if needed
- the previous/next order and `rel=prev`/`rel=next` on its neighbours
- the sheet count everywhere it appears (`grep -rn "/ 10" site`): footer title blocks, breadcrumbs, bylines and card chips
- `CARDS` in `scripts/build_social_cards.py`, then `make cards`
- the surface register in `DPP_IMPLEMENTATION.md`

For a removed or renamed URL, add a redirect in `vercel.json`. The retired `zachshallbetter.github.io/cuda-compatibility` pages redirect from the `pages-redirect` branch; never delete it.

## 6. Verification matrix

| # | Pass | How | When |
|---|---|---|---|
| 6.1 | Static checks and tests | `make site` | Every change |
| 6.2 | CI | the three PR checks | Every PR |
| 6.3 | Live audit | `make live-audit` after the merge | Every deploy |
| 6.4 | Rendering | A local server (a temporary `.claude/launch.json` entry, reverted afterwards) and the browser at 375 and 1280 px. A copy with scripts stripped and `<noscript>` unwrapped shows the no-JS view. Print with headless Chrome `--print-to-pdf` and read the PDF | Every visible change |
| 6.5 | Accessibility | The browser's accessibility tree (`read_page`) for landmark order and names. A script that focuses every control and checks for a visible focus style. Heading levels. Duplicate link names. A rendered contrast pass when colours change | Every structural or colour change |
| 6.6 | Share previews | [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/) for every changed page. It must show the page's own card, title, author and date, with no warnings. Facebook's [Batch Invalidator](https://developers.facebook.com/tools/debug/sharing/batch/) with every changed URL; each must report No Issues. For X there is no preview short of a real post, which needs the owner's approval | Title, description or card changes |
| 6.7 | Screen reader | VoiceOver (macOS or iOS) or NVDA, read by a person: skip link, landmarks, one sheet top to bottom, a table, the previous/next links | Before a release, and after structural changes |
| 6.8 | Search | Google Search Console and Bing Webmaster Tools: sitemap state, indexed pages, enhancement reports. Resubmit the sitemap after new pages, and request indexing for each new page (about four to ten a day). Google's [Rich Results Test](https://search.google.com/test/rich-results) on the home page, one sheet and one paper; all items must be valid | Monthly, after new pages, and after schema changes |

## 7. Records

After a change ships:
- Append an evidence record to `.agents/records/evidence.jsonl`: artifact, environment, procedure, result, limitations, `work_item_id`. State what was **not** verified as a limitation.
- Update the affected rows of the surface register in `DPP_IMPLEMENTATION.md`, and add a row for any new surface.
- Update the work item's progress line in `docs/WORK_ITEMS.md`, or open a new item for new scope.
- Run `make site` again (the context lock covers the records) and ship the records in a PR of their own or in the same one.

## 8. Standing duties

| Duty | Due |
|---|---|
| Renew `security.txt` `Expires` (one year ahead) | Before 2027-11-03; `make check` fails on that date |
| Bump `softwareVersion`, the footer `Project version`, `CITATION.cff` and the Homebrew formula together | Every release |
| Review the Search Console and Bing sitemap and indexing state | Monthly |
| Re-run LinkedIn Post Inspector on every page | After any card or description change |
| Re-run the attribution pilot (`docs/research/attribution-pilot/`) | After discovery changes have had time to be recrawled |

## 9. Open items

These are known and not yet done. Close each one by doing it and recording evidence (§7).

1. **Screen-reader session (6.7).** Not yet run. The accessibility tree and focus pass were checked, but announcements were never heard.
2. **X card layout.** X lays the page title over the bottom-left of the share image, covering the card's URL line (checked 2026-11-03 with a real post). Facebook (Batch Invalidator: all 18 pages, no issues) and LinkedIn (all 18, no warnings) are done. Facebook's only note is the optional `fb:app_id`, which needs a Meta app.
3. **Indexing requests.** Google's daily quota ran out after four of the eight papers (2026-11-03). Request indexing for `runtime-version-facade`, `implementation-gap-audit-2026-11`, `attribution-pilot` and `cuda-12-9-to-13-0-changes` in Search Console. The resubmitted sitemap already lists all 18 pages.
4. **Full contrast re-measure.** The 1,943-node pass predates the dark-on-light change. Colours were judged per section since then.
5. **The generator branch.** `feat/generated-site` holds an unpushed site generator from before CLC-035. Either port today's page contract into it or retire it. Until then, pages are edited by hand under this protocol.
