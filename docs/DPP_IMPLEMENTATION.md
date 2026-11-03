# Digital Presence Implementation Record

**Protocol:** Digital Presence Protocol 0.3.0 Working Draft
**Pinned revision:** `38eb016986731bed3eaa7aecfc6ba14f3fbc8ccd`
**Scope:** CUDA Compatibility project website, source repository, and linked public identity surfaces
**Last reviewed:** 2026-11-03
**Standing:** Partial local implementation; no DPP conformance or external reach claim

This record applies the pinned DPP objects and evidence rules to the project’s public presence. It is an implementation ledger, not a guarantee of search visibility, AI citations, or user outcomes. The DPP protocol set itself is a proposed specification and does not declare its own implementation conformant.

## Entity and identity

| ID | Entity | Canonical identifier | Controller and boundary |
|---|---|---|---|
| `cuda-compatibility` | Open-source CUDA compatibility project (`SoftwareApplication`) | `https://cudacompat.com/#software` | Project maintained by Zach Shallbetter; NVIDIA is not affiliated. |
| `cuda-compatibility-site` | Project website (`WebSite`) | `https://cudacompat.com/#website` | Project-controlled source served by Vercel at the apex `cudacompat.com`; availability is a separate state. |
| `zach-shallbetter` | Project author (`Person`) | `https://cudacompat.com/#author` | Author identity links to `https://zachshallbetter.com/` and `https://github.com/zachshallbetter`; those external surfaces are not controlled by this site. The homepage's About the author note and the `Person` markup carry the same facts: systems engineer and independent researcher, Spokane, Washington, practising through Snack Studio (`worksFor`, `https://www.snackstud.io/`), which is separate from the project. |

The homepage is the entity’s canonical description. Documentation pages are scoped representations of the same project and link their `TechArticle` and `BreadcrumbList` metadata to the canonical site and author IDs. The repository URL is `https://github.com/zachshallbetter/cuda-compatibility`.

Each public HTML page is a surface identified by its canonical URL. Its article representation uses the stable `canonical-URL#article` identifier where the page has `TechArticle` markup; the homepage uses the stable `#software`, `#website`, and `#author` entity IDs. [`scripts/site_checks.py`](../scripts/site_checks.py) enforces these links.

## Claim and source register

| ID | Public claim | Qualification and source of truth | Current expression |
|---|---|---|---|
| `CLC-CLAIM-001` | The project is independent research, not affiliated with NVIDIA. | Project identity and licensing; [`.agents/project/intent.md`](../.agents/project/intent.md), [`LICENSE-MIT`](../LICENSE-MIT), [`LICENSE-APACHE`](../LICENSE-APACHE). | Homepage attribution and documentation. |
| `CLC-CLAIM-002` | The current launcher targets Linux x86_64 and visible `sm_52` devices. | Implementation contract; launcher and profile code plus [architecture](architecture.md). | Homepage, install guide, compatibility guide. |
| `CLC-CLAIM-003` | The process-local facade reports CUDA Runtime 13.0 for one query while retaining the actual GPU identity. | Reproduction record; [runtime-version facade research](research/runtime-version-facade.md). This is not general CUDA 13 runtime support. | Homepage architecture and feature sections. |
| `CLC-CLAIM-004` | Selected PyTorch operation cases have native Maxwell paths. | Operation registration and test receipts in the [PyTorch research record](research/pytorch-cu130-maxwell.md); each listed dtype/shape limit and per-card result remains part of the claim. | Homepage and support matrix. |
| `CLC-CLAIM-005` | Selected PyTorch operations can use bounded CPU fallback; one small first-order training slice exercises selected differentiable fallback routes. | Registration, error classification, gradient evidence, and limits in [`torch_adapter.py`](../src/cuda_legacy_compat/torch_adapter.py) and the [PyTorch research record](research/pytorch-cu130-maxwell.md). This does not establish GPU convolution or general training support. | Homepage and support matrix. |
| `CLC-CLAIM-006` | GM200 `sm_52` reference devices have run selected checks; other catalog cards are unverified. | Hardware run evidence in [work items](WORK_ITEMS.md) and research records; NVIDIA’s legacy table supports model-to-capability mapping only. | Homepage hardware table and GPU catalog. |
| `CLC-CLAIM-007` | Current PyTorch CUDA 13 builds and older GPU support have separate package and hardware constraints. | Version-pinned PyTorch release matrix and NVIDIA CUDA 13 release notes linked from [compatibility targets](compatibility-targets.md). | Compatibility guide and homepage. |
| `CLC-CLAIM-008` | Broad CUDA API/library coverage, general training/autograd, performance parity, and machine-wide interposition are not established. | Bounded by source registrations and current test scope; one tiny first-order training path does not establish arbitrary autograd or optimizer support. Absence of a registered path is not evidence that an operation is impossible. | Homepage, compatibility guide, and research page. |

**Public example.** [`examples/sm52-mlp`](../examples/sm52-mlp/README.md) gives one recoverable path for `CLC-CLAIM-002`, `004`, `005` and `008`: the pinned release, hardware, driver, framework and toolchain; commands; expected output; per-operation CPU comparison and route; and two documented unsupported cases. Its reproduction record is `GH-50-public-example-sm52-mlp`; outside runs are reported through the [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml) issue form.

All claims are project-authored and are not independent corroboration. CUDA/package requirements cite primary vendor or framework sources. Test claims point to named local procedures and results; a repeated site statement does not upgrade its evidence.

For vendor claims, the publisher and immutable release/version are recorded in `.agents/project/formal-resources.json` and the corresponding compatibility/research page. For project results, the named research record supplies the procedure and measured scope. The project author is also the implementer, so conflict-of-interest classification is **self-authored project evidence**; independent review is not claimed. If a source date, independent evaluator, or observation method is unavailable, it remains unknown rather than inferred.

## Representation and surface register

The state dimensions are evaluated independently. `SATISFIED` below means the named local criterion passed; it does not imply public reach.

| Surface ID | Representation | Represented | Accessible | Discoverable | Observed | Engaged | Outcome |
|---|---|---|---|---|---|---|
| `site-home` | Homepage, canonical `/` | `SATISFIED` locally | Public HTTP `SATISFIED` (200 on `cudacompat.com`; 2026-11-03) | `SATISFIED` as linked, sitemap-listed, crawl-allowed content; indexing `GRAY` | Public fetch `SATISFIED`; crawler/search observation `GRAY` | Consented GA4 events defined; no audience data | `GRAY` |
| `site-docs` | User guides under `/docs/` | `SATISFIED` locally | Public HTTP `SATISFIED` (8 canonical routes; 200; 2026-11-03) | `SATISFIED` as linked, sitemap-listed, crawl-allowed content with visible breadcrumbs matching `BreadcrumbList`; indexing `GRAY` | Public fetch `SATISFIED`; search state `GRAY` | `GRAY` | `GRAY` |
| `site-gpu-catalog` | Explained NVIDIA legacy compute-capability catalog | `SATISFIED` locally | Public HTTP `SATISFIED` (200; 2026-11-03) | `SATISFIED` as internally linked and sitemap-listed; indexing `GRAY` | Public fetch `SATISFIED`; search state `GRAY` | `GRAY` | `GRAY` |
| `site-robots` | `/robots.txt` crawler policy | `SATISFIED` in source | Public fetch `SATISFIED` (200; 2026-11-03) | `SATISFIED` as a published route; crawler compliance `GRAY` | Public bytes fetched; no crawler receipt | `NOT_APPLICABLE` | `NOT_APPLICABLE` |
| `site-sitemap` | `/sitemap.xml` with truthful `lastmod` | `SATISFIED` in source | Public fetch `SATISFIED` (200; 2026-11-03) | `SATISFIED` as a published canonical URL list; search discovery/indexing `GRAY` | All 10 listed pages returned 200; no search-engine receipt for `cudacompat.com` | `NOT_APPLICABLE` | `NOT_APPLICABLE` |
| `site-agent-index` | `/llms.txt` | `SATISFIED` in source | Public fetch `SATISFIED` (200; 2026-11-03) | `SATISFIED` as an informal published index; tool adoption `GRAY` | Public bytes fetched; no agent retrieval trace | `GRAY` | `GRAY` |
| `site-agent-fulltext` | `/llms-full.txt`, the text of every sitemap page as Markdown | `SATISFIED` in source; regenerated by `scripts/build_llms_full.py` and rejected by `make check` when stale | Not yet deployed at review time | `SATISFIED` as linked from `llms.txt` and the discovery sheet; adoption `GRAY` | No agent retrieval trace | `GRAY` | `GRAY` |
| `site-share-assets` | Favicon set (ICO 16/32/48, SVG, mask), touch and manifest icons, and a 1200×630 social card per page | `SATISFIED` in source; sizes, links and per-page cards enforced by `make check` | Local fetch only at review time | `GRAY` (platform previews unobserved) | `GRAY` | `NOT_APPLICABLE` | `NOT_APPLICABLE` |
| `site-not-found` | `/404.html` error page | `SATISFIED` in source (`noindex`, outside the sitemap; enforced by `make check`) | Not yet deployed at review time | `NOT_APPLICABLE` | `GRAY` | `NOT_APPLICABLE` | `NOT_APPLICABLE` |
| `distribution-homebrew` | `Formula/cuda-legacy.rb`, a Linux-only formula for release `v0.1.0` (with an optional HEAD build), tapped from this repository | `SATISFIED` in source (`brew style` clean; equivalent virtualenv install exercised on macOS) | Release archive checksum recorded; `brew install` not run on Linux | `GRAY` | `GRAY` | `GRAY` | `GRAY` |
| `source-repository` | GitHub repository URL | `SATISFIED` as a public source repository | Public repository and default branch resolve; exact branch updates are recorded in `.agents/records/evidence.jsonl`. | GitHub surface exists; repository discovery `GRAY` | GitHub API confirms the public repository; ACP notice reports governed/normal | `GRAY` | `GRAY` |
| `author-site`, `author-github` | Author homepage and profile links | `SATISFIED` as site references | `GRAY` | `GRAY` | No independent check in this review | `GRAY` | `GRAY` |

Local representations are versioned in Git and their selected source hashes are recorded by `.agents/context-lock.json`. There is no deployed response snapshot or per-platform rendered hash for this candidate.

### Current deployment — 2026-11-03 (Vercel)

The site is served by the Vercel project `cuda-compatibility` from `main` with Root Directory `site`. The owner made the apex `cudacompat.com` the primary domain; `www.cudacompat.com` returns 308 to it, and the former `cuda-compatibility.vercel.app` alias was removed. All 26 audited routes and assets returned HTTP 200 at 2026-11-03T10:53:12Z after the ACP-authorized deploy of `28cd261dab8ab0832db42864fd6f2d426eb5bcf8` (evidence `CLC-030-apex-route-audit` and the PR 61 ACP records). The GitHub Pages observations below are history; that host is retired by intent but still serves until the owner disables it.

### Accessibility observations — 2026-11-03 (local preview)

Measured in a local preview, not on the public origin: every page has one `h1`, no skipped heading levels, `lang`, labelled landmarks, a skip link, alt text or `aria-hidden` on every image and SVG, accessible names on all links, buttons and tables, and no horizontal scroll at 320, 375 and 768 px. A rendered contrast pass over 1,943 text nodes found no text below WCAG 2.2 AA after removing inline label opacity and two badge colour conflicts; the only flagged element was a button caught mid-transition between two passing states. Every standalone link and button is at least 24 px tall. Status marks combine shape and text, so no state depends on colour alone. Assistive-technology sessions with a screen reader have not been run.

### Public deployment observation — 2026-11-01 (retired GitHub Pages)

The repository update from `22667016a0473178f8eea37fcaa8bb876f3c3a67` to `b46c42bb4209842cea3825874c12225e6f2e3f50` on `codex/cuda-compat-publication` received exact ACP `ALLOW` request `7d837ae7ae30e83a`. GitHub Actions run [41309056623](https://github.com/zachshallbetter/cuda-compatibility/actions/runs/41309056623) completed successfully with `headSha=b46c42bb4209842cea3825874c12225e6f2e3f50`, after exact production-deploy ACP `ALLOW` request `606588308982667e`. The deployed canonical site returned HTTP 200. A direct audit fetched all 10 sitemap HTML URLs, `robots.txt`, `sitemap.xml`, `llms.txt`, and PNG/SVG social cards; all 15 returned HTTP 200. All nine project-source links discovered on the published documentation pages also returned HTTP 200 and pointed at the published branch. The live homepage included canonical, Open Graph, Twitter, and structured-data metadata. These observations establish source publication, deployment, availability, and served metadata only; they do not establish indexing, crawler use, search visibility, social-card rendering by third parties, agent retrieval, or audience outcomes. The full response-size and route audit is recorded in `.agents/records/evidence.jsonl`.

### Prior deployment observation — 2026-11-01 (retired GitHub Pages)

Commit `755c6d6cbcea32f886b24fe91f4a48f28fe0bb11` was pushed to `codex/cuda-compat-publication` after exact ACP `ALLOW` request `29b29fb9b2606225`; Pages workflow [41311091751](https://github.com/zachshallbetter/cuda-compatibility/actions/runs/41311091751) completed successfully at that exact SHA after deploy request `a817646862555bc1`. The live audit returned HTTP 200 for all 10 sitemap pages, `sitemap.xml`, `robots.txt`, `llms.txt`, and both social-card assets. Exact response sizes and URLs are recorded under `CLC-026-publication-755c6d6` in `.agents/records/evidence.jsonl`. Search indexing, AI retrieval, social-card rendering by platforms, and audience outcomes remain unobserved.

### Final GitHub Pages state — 2026-11-01 (retired)

Commit `bbcb5f74b55f183f9e5dd702352341f26a7b65b4` is the latest verified **site deployment**. GitHub Pages run [41313898493](https://github.com/zachshallbetter/cuda-compatibility/actions/runs/41313898493) completed successfully at that exact SHA after ACP push request `c0cc923cab37342d` and deploy request `6dff38b12237eb4e`; both were exact `ALLOW` decisions and capability checks passed. A fresh public audit returned HTTP 200 for all 10 sitemap pages, the sitemap, `robots.txt`, `llms.txt`, and three supporting assets (16/16 total); the 10 live descriptions are unique and 141–167 characters. The source update from `f31d06e79439353a0da42a5f50c7f02e85d13579` to `5f54600c54f1a131101e1232e069ad7943c9f117` was published after exact ACP `ALLOW` request `404c5e16d4a5859f` and capability verification. It publishes the official research charter, product contract, milestone plan, and reconciled release record; it changes no site files, so the deployed site remains `bbcb5f7`. The consent-gated GA4 tag is deployed; one owner-initiated accepted test session produced `page_view`, `first_visit`, and `session_start`, then consent was withdrawn. Rejection-path network inspection observed no Google measurement requests. Google Search Console ownership is verified by HTML tag; its sitemap report says “Couldn't fetch” with zero pages, and URL Inspection reports the homepage is unknown to the index with no referring sitemap. A live URL test says the homepage is available to Google and can be indexed; an indexing request was not submitted because the console presented a CAPTCHA. Bing reports verified administrator access by meta-tag authentication; its sitemap reports Success with 10 URLs discovered and zero errors or warnings. No indexing, AI retrieval, ranking, or audience outcome is claimed. Exact release, webmaster, and route evidence is in `.agents/records/evidence.jsonl`.

The local preview origin is a development surface, not a public or production surface. Do not cite its HTTP responses as evidence of public accessibility.

## Discovery policy and query-intent corpus

The site’s durable user jobs are compatibility-by-version, compatibility-by-card, install-and-run, error diagnosis, operation coverage, architecture, and evidence review. The following **query corpus v0.1.0** is a manually authored coverage seed for the CUDA Compatibility project, in English for the `en-US` locale. It has no collected sample frame, query frequency, Google Trends scores, live search results, ranking observations, or representative-demand claim. It is a hypothesis list, not evidence of demand.

| Query ID | Query family / brand | Exact query / prompt | Intent | Expected answer facts and evidence destination | Status |
|---|---|---|---|---|---|
| `Q-CUDA13-001` | Compatibility evaluation · unbranded | Can I run CUDA 13 PyTorch on an older NVIDIA GPU? | Feasibility | Experimental; selected PyTorch paths only; Linux x86_64 and `sm_52` gate; no blanket compatibility. Source: `site/docs/compatibility.html`. | Defined; not queried live |
| `Q-CUDA13-002` | Toolkit policy · unbranded | Does CUDA 13 support Maxwell? | Information | NVIDIA’s CUDA 13 support change is separate from GPU capability and does not mean all Maxwell computation is impossible. Source: `docs/compatibility-targets.md`; CUDA 13 release notes. | Defined; not queried live |
| `Q-CUDA13-003` | Card lookup · unbranded | Can a GTX 980 Ti run a CUDA 13 PyTorch wheel? | Specific-card evaluation | NVIDIA lists the model at `sm_52`; it is an unverified project candidate, not a certified configuration. Source: `site/gpu-catalog.html`; NVIDIA legacy GPU list. | Defined; not queried live |
| `Q-CUDA13-004` | Error diagnosis · unbranded | What does “no kernel image is available” mean in PyTorch? | Troubleshooting | The loaded binary has no usable image for the device; version spoofing alone cannot supply a kernel. Check the exact op, library, dtype, and shape. Source: architecture and compatibility guides. | Defined; not queried live |
| `Q-CUDA13-005` | Installation · unbranded | How do I install a CUDA compatibility layer for an old GPU? | Setup | Install into the same Python environment as PyTorch; satisfy the documented Linux/toolchain requirements; use the opt-in launcher. Source: `site/docs/getting-started.html`. | Defined; not queried live |
| `Q-CUDA13-006` | Operation lookup · unbranded | Which PyTorch CUDA 13 operations work on `sm_52`? | Support lookup | Report only registered native and CPU-fallback cases with exact limits; unknown combinations fail explicitly. Source: `site/docs/compatibility.html`. | Defined; not queried live |
| `Q-CUDA13-007` | Training capability · unbranded | Can I train or use autograd on Maxwell with this tool? | Capability boundary | A small float32 linear→ReLU→sum model has first-order gradient evidence for seven SGD steps on GPUs 1 and 2; broad training and arbitrary autograd paths remain unverified. Source: compatibility and research pages. | Defined; not queried live |
| `Q-CUDA13-008` | Performance evaluation · unbranded | Is CPU fallback as fast as GPU execution? | Performance evaluation | No parity claim; transfers and CPU execution can dominate, and broad performance data are unavailable. Source: compatibility and research pages. | Defined; not queried live |

Before any trend or ranking claim, capture the provider, date range, geography, category, query variants, and raw export; do not substitute this intent corpus for those observations.

## Measurement dictionary and current observations

Each dimension is evaluated separately. Empty or unavailable data is `GRAY`, never zero.

| Metric ID | Dimension | Definition and denominator | Source / window | Current state |
|---|---|---|---|---|
| `DPP-AV-01` | Availability | Successful authorized URL fetches / attempted eligible canonical URLs; report response class per URL. | Public fetch of `cudacompat.com` on 2026-11-03. | 26/26 routes and assets returned 200 (10 sitemap pages plus 16 discovery, style, script, font and model files). |
| `DPP-ER-01` | Entity recognition | Correct project/author resolution / scored entity opportunities, including ambiguous cases. | Labeled manual query/agent sample, versioned per run. | No observations; `GRAY`. |
| `DPP-RT-01` | Retrieval | Directly traced source retrievals / instrumented retrieval opportunities. A cited or mentioned URL without a trace is an answer mention, not verified retrieval. | Instrumented retrieval logs; provider/model/time required. | No trace access; `GRAY`. |
| `DPP-SF-01` | Semantic fidelity | Supported expected facts preserved / evaluable expected facts in sampled answers. | Human rubric against claim register and versioned query corpus. | No answer sample; `GRAY`. |
| `DPP-EQ-01` | Evidence qualification | Correct support/uncertainty classifications / adjudicated claim rulings. | Human review against claim/source register. | No independent answer rulings; `GRAY`. |
| `DPP-AT-01` | Attribution | Correct source and author attributions / attributable answers. | Manual source/author review of answer samples. | No answer sample; `GRAY`. |
| `DPP-PC-01` | Propagation | Observed eligible surfaces carrying a claim / monitored eligible surfaces. This is monitored coverage, not a census. | Versioned owned/earned/partner surface inventory. | No campaign or monitoring set; `GRAY`. |
| `DPP-FR-01` | Freshness | Time from a recorded correction to first observed corrected representation; uncorrected cases are right-censored. | Dated correction and crawl receipts with sampling interval. | No correction observations; `GRAY`. |
| `DPP-CO-01` | Conversion | Incremental qualified outcomes / eligible randomized or matched exposure units; observed click-through is reported separately. | Requires consented, privacy-reviewed event data and a defensible counterfactual. | No conversion tracking or baseline; `GRAY`. |
| `DPP-BI-01` | Boundary integrity | Passing authorized harmless-fixture trials / conclusive authorized trials. | Separate security-test scope and tool traces; never mixed with visibility scores. | No DPP-ASI boundary experiment; `NOT_APPLICABLE` to current site release. |
| `DPP-SEARCH-01` | Search visibility | Search Console impressions, clicks, indexed URL sample, and query/page dimensions; position is not a universal rank. | Google Domain property `sc-domain:cudacompat.com` and Bing site `https://cudacompat.com/`, observed 2026-11-03 (evidence `GH-52-discovery-baseline`). | Google: sitemap Success, 10 discovered pages; homepage indexed (smartphone crawl 2026-11-03, Google-selected canonical equals the declared one); other nine pages `GRAY`; performance data still processing. Bing: sitemap Success, 10 URLs discovered. "Search generative AI" is set to Include and was not changed. IndexNow accepted 10 URLs (HTTP 202). |
| `DPP-ANALYTICS-01` | Consented site activity | Allowlisted events only after affirmative consent: page views, classified link clicks with page area, source downloads, successful command copies, hero replays; report opt-in coverage and missingness. | GA4 property `562082064`, web stream `16261288778` (`G-Q7C82JK9GM`); enhanced measurement and automatic detection off, signals and ads personalisation off, two-month retention, owner traffic excluded. | Events traced locally (`CLC-031-browser-event-trace`); no audience data yet; `GRAY`. |

The project does not currently have field performance data, search-index coverage, AI answer observations, or a Search Console baseline. GA4 is configured and deployed behind explicit consent. One owner-initiated test session produced `page_view`, `first_visit`, and `session_start` in Realtime after consent; the test consent was then withdrawn. This verifies event receipt only, not audience coverage or outcomes. Property controls have Enhanced Measurement, Google Signals, user-provided data collection, and ads personalization off; event retention is two months and configured redaction covers email/token/code query keys. The local site checker reports implementation properties, not audience outcomes.

## Distribution and intervention ledger

Classes: **owned** (project-controlled surfaces), **earned** (third-party destinations that choose whether to carry it), **partner** (an agreed placement with another party), **paid**, and **algorithmic** (reach decided by a platform's ranking, such as a social feed or search). One placement can be owned in source and algorithmic in reach; both are listed. Drafts and their states live in [`docs/publishing/`](publishing/README.md). This table records placements only.

| Activity | Class | State | Authority and receipt |
|---|---|---|---|
| Local website and source changes | Owned | Pre-publication | Local work items and `.agents/records/evidence.jsonl`; reversible in source control. |
| GitHub repository | Owned | Published | Public repository `zachshallbetter/cuda-compatibility`; each `main` update follows an exact ACP `ALLOW` recorded in `.agents/records/evidence.jsonl`. |
| GitHub Pages site | Owned | Retired | Disabled by the owner on 2026-11-03; the former address returns 404. |
| https://cudacompat.com/ | Owned | Live | Vercel production deploys from `main` after exact ACP `ALLOW`; deployed revisions are in `.agents/records/evidence.jsonl`. |
| Homebrew formula and release | Owned | Published | GitHub release `v0.1.0` (tag `3c2cf3d64ce6a0adc82baf789a9b31e7c8b19a3b` on commit `de5925497c215f1382321f3a57cc7097f6e30f8c`) after exact ACP `ALLOW` `7871607dcb298b4c`; the formula pins its archive by SHA-256. |
| Public example and reproduction-report form | Owned | Published | `examples/sm52-mlp/` and the issue form, merged after ACP `ALLOW` `a48f0bbbd8a62a1f`; evidence `GH-50-public-example-sm52-mlp`. |
| Search engines (Google, Bing, IndexNow) | Algorithmic | Submitted | Sitemaps report Success (Google 10 pages, Bing 10 URLs); the homepage is indexed in Google; IndexNow accepted 10 URLs (HTTP 202). Submission and indexing do not show ranking or reach. Evidence: `DPP-SEARCH-01`. |
| Article, short videos, LinkedIn, X | Owned profiles, algorithmic reach | Drafted, not posted | [`docs/publishing/`](publishing/README.md); every draft is in the Draft state. |
| Hacker News, Lobsters, PyTorch Forums, Reddit, publications | Earned | Not posted | Shortlist and eligibility in [`docs/publishing/channels.md`](publishing/channels.md). Show HN is on hold until outside reproduction. |
| Partner placements | Partner | None | No agreements. |
| Paid promotion | Paid | None | Not budgeted; out of scope without separate authority. |

Every future placement must record the versioned source asset, destination, method, audience, rights and permission basis, owner, publication receipt (URL, timestamp and exact text), and resulting observations. No automated or mass outreach is part of this project.

## Module coverage and release gates

| DPP module | Implementation level | Evidence and unresolved work |
|---|---|---|
| PRP — representation | Partial, locally implemented | Stable project/site/author IDs, page canonicals, authorship, visible claim/source map, and correction triggers are recorded. Cross-platform identity reconciliation is `GRAY`. |
| DAP — discovery/access | Local controls implemented and public availability observed; search/agent discovery `GRAY` | HTML, breadcrumbs, internal links, sitemap with `lastmod`, robots, `llms.txt`, `llms-full.txt`, `noindex` 404, security headers and an IndexNow key pass local checks; 26/26 public routes returned 200 before this revision. Accessibility measured locally (see above). Indexation, social rendering, crawler use, and agent retrieval remain `GRAY`. |
| DIS — distribution | Drafts prepared; no campaign | Owned, earned, partner, paid and algorithmic placements are classified in the ledger; drafts are in `docs/publishing/`; no social, earned, partner or paid placement is recorded. |
| AEP — evidence | Partial, locally implemented | Material technical claims map to primary documentation, implementation records, or named tests. No independent expert review or external corroboration is claimed. |
| OIP — observation | Metrics defined; observations unavailable | Metric definitions, sources, denominators, and blind spots are recorded; no platform baseline is available. |
| OEP — experimentation | Not implemented for presence outcomes | No live SEO, crawler, content, or AI-citation experiment is registered. Site edits and local regression checks are not audience-effect experiments. |
| MPF — measurement/provenance | Partial documentation and a validation adapter; not conformant | Seed query corpus and metric dictionary exist. Observation records are checked by `scripts/dpp_measurement.py` against the vendored pinned schema plus project semantic rules ([measurement validation](dpp/measurement-validation.md)); one real record exists in `examples/dpp/`. Raw external observations at scale, retrieval traces, adjudications, uncertainty estimates, and preregistration receipts do not. |
| ASI — AI surfaces | Not implemented | The site publishes server-rendered HTML, an informal `llms.txt` and a generated `llms-full.txt`, and declares selected crawler preferences. It has no instrumented retrieval pipeline or verified answer/citation sample. |
| Release gates | Vercel deployment verified per revision; external reach gates tracked separately | `scripts/site_checks.py` (canonical, metadata, links, sitemap, robots, agent index, full-text freshness, event allowlist, 404 `noindex`), `make check`, and `make test` pass; each production deploy is preceded by exact ACP `ALLOW`s and followed by a live route audit. |

No DPP module is declared conformant. A validator pass, sitemap entry, crawler allowance, answer mention, citation, or local preview cannot be upgraded into an indexing, reach, or causal-outcome claim.

## Maintenance and correction triggers

Review this record when CUDA/PyTorch/tool versions or supported operations change; a material source claim changes; a canonical URL, crawler policy, privacy practice, or analytics configuration changes; the public site is deployed; or new Search Console, analytics, crawler, AI-answer, or distribution evidence is collected. Preserve prior observations and record corrections with their affected claim and representation IDs.
