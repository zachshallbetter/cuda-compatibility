# Digital Presence Implementation Record

**Protocol:** Digital Presence Protocol 0.3.0 Working Draft
**Pinned revision:** `38eb016986731bed3eaa7aecfc6ba14f3fbc8ccd`
**Scope:** CUDA Compatibility project website, source repository, and linked public identity surfaces
**Last reviewed:** 2026-11-01
**Standing:** Partial local implementation; no DPP conformance or external reach claim

This record applies the pinned DPP objects and evidence rules to the project’s public presence. It is an implementation ledger, not a guarantee of search visibility, AI citations, or user outcomes. The DPP protocol set itself is a proposed specification and does not declare its own implementation conformant.

## Entity and identity

| ID | Entity | Canonical identifier | Controller and boundary |
|---|---|---|---|
| `cuda-compatibility` | Open-source CUDA compatibility project (`SoftwareApplication`) | `https://zachshallbetter.github.io/cuda-compatibility/#software` | Project maintained by Zach Shallbetter; NVIDIA is not affiliated. |
| `cuda-compatibility-site` | Project website (`WebSite`) | `https://zachshallbetter.github.io/cuda-compatibility/#website` | Project-controlled source; GitHub Pages availability is a separate state. |
| `zach-shallbetter` | Project author (`Person`) | `https://zachshallbetter.github.io/cuda-compatibility/#author` | Author identity links to `https://zachshallbetter.com/` and `https://github.com/zachshallbetter`; those external surfaces are not controlled by this site. |

The homepage is the entity’s canonical description. Documentation pages are scoped representations of the same project and link their `TechArticle` and `BreadcrumbList` metadata to the canonical site and author IDs. The repository URL is `https://github.com/zachshallbetter/cuda-compatibility`.

Each public HTML page is a surface identified by its canonical URL. Its article representation uses the stable `canonical-URL#article` identifier where the page has `TechArticle` markup; the homepage uses the stable `#software`, `#website`, and `#author` entity IDs. `scripts/site_checks.py` enforces these links.

## Claim and source register

| ID | Public claim | Qualification and source of truth | Current expression |
|---|---|---|---|
| `CLC-CLAIM-001` | The project is independent research, not affiliated with NVIDIA. | Project identity and licensing; `PROJECT_INTENT.md`, `LICENSE-MIT`, `LICENSE-APACHE`. | Homepage attribution and documentation. |
| `CLC-CLAIM-002` | The current launcher targets Linux x86_64 and visible `sm_52` devices. | Implementation contract; launcher and profile code plus `docs/architecture.md`. | Homepage, install guide, compatibility guide. |
| `CLC-CLAIM-003` | The process-local facade reports CUDA Runtime 13.0 for one query while retaining the actual GPU identity. | Reproduction record; `docs/research/runtime-version-facade.md`. This is not general CUDA 13 runtime support. | Homepage architecture and feature sections. |
| `CLC-CLAIM-004` | Selected PyTorch operation cases have native Maxwell paths. | Operation registration and test receipts in `docs/research/pytorch-cu130-maxwell.md`; each listed dtype/shape limit and per-card result remains part of the claim. | Homepage and support matrix. |
| `CLC-CLAIM-005` | Selected PyTorch operations can use bounded CPU fallback; one small first-order training slice exercises selected differentiable fallback routes. | Registration, error classification, gradient evidence, and limits in `src/cuda_legacy_compat/torch_adapter.py` and the PyTorch research record. This does not establish GPU convolution or general training support. | Homepage and support matrix. |
| `CLC-CLAIM-006` | GM200 `sm_52` reference devices have run selected checks; other catalog cards are unverified. | Hardware run evidence in `docs/WORK_ITEMS.md` and research records; NVIDIA’s legacy table supports model-to-capability mapping only. | Homepage hardware table and GPU catalog. |
| `CLC-CLAIM-007` | Current PyTorch CUDA 13 builds and older GPU support have separate package and hardware constraints. | Version-pinned PyTorch release matrix and NVIDIA CUDA 13 release notes linked from `docs/compatibility-targets.md`. | Compatibility guide and homepage. |
| `CLC-CLAIM-008` | Broad CUDA API/library coverage, general training/autograd, performance parity, and machine-wide interposition are not established. | Bounded by source registrations and current test scope; one tiny first-order training path does not establish arbitrary autograd or optimizer support. Absence of a registered path is not evidence that an operation is impossible. | Homepage, compatibility guide, and research page. |

All claims are project-authored and are not independent corroboration. CUDA/package requirements cite primary vendor or framework sources. Test claims point to named local procedures and results; a repeated site statement does not upgrade its evidence.

For vendor claims, the publisher and immutable release/version are recorded in `FORMAL_RESOURCE_MANIFEST.json` and the corresponding compatibility/research page. For project results, the named research record supplies the procedure and measured scope. The project author is also the implementer, so conflict-of-interest classification is **self-authored project evidence**; independent review is not claimed. If a source date, independent evaluator, or observation method is unavailable, it remains unknown rather than inferred.

## Representation and surface register

The state dimensions are evaluated independently. `SATISFIED` below means the named local criterion passed; it does not imply public reach.

| Surface ID | Representation | Represented | Accessible | Discoverable | Observed | Engaged | Outcome |
|---|---|---|---|---|---|---|
| `site-home` | Homepage, canonical `/` | `SATISFIED` locally | Public HTTP `SATISFIED` (200; 2026-11-01) | `SATISFIED` as linked, sitemap-listed, crawl-allowed content; indexing `GRAY` | Public fetch `SATISFIED` (200); crawler/search observation `GRAY` | `GRAY` | `GRAY` |
| `site-docs` | User guides under `/docs/` | `SATISFIED` locally | Public HTTP `SATISFIED` (8 canonical routes; 200; 2026-11-01) | `SATISFIED` as linked, sitemap-listed, crawl-allowed content; indexing `GRAY` | Public fetch `SATISFIED` (all 8 routes returned 200); crawler/search observation `GRAY` | `GRAY` | `GRAY` |
| `site-gpu-catalog` | Explained NVIDIA legacy compute-capability catalog | `SATISFIED` locally | Public HTTP `SATISFIED` (200; 2026-11-01) | `SATISFIED` as internally linked and sitemap-listed; indexing `GRAY` | Public fetch `SATISFIED` (200); search state `GRAY` | `GRAY` | `GRAY` |
| `site-robots` | `/robots.txt` crawler policy | `SATISFIED` in source | Public fetch `SATISFIED` (200; 2026-11-01) | `SATISFIED` as a published route; crawler compliance `GRAY` | Public bytes fetched; no crawler receipt | `NOT_APPLICABLE` | `NOT_APPLICABLE` |
| `site-sitemap` | `/sitemap.xml` | `SATISFIED` in source | Public fetch `SATISFIED` (200; 2026-11-01) | `SATISFIED` as a published canonical URL list; search discovery/indexing `GRAY` | Public sitemap fetched and its 10 listed pages returned 200; search-engine processing `GRAY` | `NOT_APPLICABLE` | `NOT_APPLICABLE` |
| `site-agent-index` | `/llms.txt` | `SATISFIED` in source | Public fetch `SATISFIED` (200; 2026-11-01) | `SATISFIED` as an informal published index; tool adoption `GRAY` | Public bytes fetched; no agent retrieval trace | `GRAY` | `GRAY` |
| `source-repository` | GitHub repository URL | `SATISFIED` as a public source repository | Public repository and default branch resolve; exact branch updates are recorded in `records/evidence.jsonl`. | GitHub surface exists; repository discovery `GRAY` | GitHub API confirms the public repository; ACP notice reports governed/normal | `GRAY` | `GRAY` |
| `author-site`, `author-github` | Author homepage and profile links | `SATISFIED` as site references | `GRAY` | `GRAY` | No independent check in this review | `GRAY` | `GRAY` |

Local representations are versioned in Git and their selected source hashes are recorded by `.agents/context-lock.json`. There is no deployed response snapshot or per-platform rendered hash for this candidate.

### Public deployment observation — 2026-11-01

The repository update from `22667016a0473178f8eea37fcaa8bb876f3c3a67` to `b46c42bb4209842cea3825874c12225e6f2e3f50` on `codex/cuda-compat-publication` received exact ACP `ALLOW` request `7d837ae7ae30e83a`. GitHub Actions run [41309056623](https://github.com/zachshallbetter/cuda-compatibility/actions/runs/41309056623) completed successfully with `headSha=b46c42bb4209842cea3825874c12225e6f2e3f50`, after exact production-deploy ACP `ALLOW` request `606588308982667e`. The deployed canonical site returned HTTP 200. A direct audit fetched all 10 sitemap HTML URLs, `robots.txt`, `sitemap.xml`, `llms.txt`, and PNG/SVG social cards; all 15 returned HTTP 200. All nine project-source links discovered on the published documentation pages also returned HTTP 200 and pointed at the published branch. The live homepage included canonical, Open Graph, Twitter, and structured-data metadata. These observations establish source publication, deployment, availability, and served metadata only; they do not establish indexing, crawler use, search visibility, social-card rendering by third parties, agent retrieval, or audience outcomes. The full response-size and route audit is recorded in `records/evidence.jsonl`.

### Latest deployment observation — 2026-11-01

Commit `755c6d6cbcea32f886b24fe91f4a48f28fe0bb11` was pushed to `codex/cuda-compat-publication` after exact ACP `ALLOW` request `29b29fb9b2606225`; Pages workflow [41311091751](https://github.com/zachshallbetter/cuda-compatibility/actions/runs/41311091751) completed successfully at that exact SHA after deploy request `a817646862555bc1`. The live audit returned HTTP 200 for all 10 sitemap pages, `sitemap.xml`, `robots.txt`, `llms.txt`, and both social-card assets. Exact response sizes and URLs are recorded under `CLC-026-publication-755c6d6` in `records/evidence.jsonl`. Search indexing, AI retrieval, social-card rendering by platforms, and audience outcomes remain unobserved.

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
| `Q-CUDA13-007` | Training capability · unbranded | Can I train or use autograd on Maxwell with this tool? | Capability boundary | A small float32 linear→ReLU→sum SGD case has first-order gradient evidence on one GPU; broad training and arbitrary autograd paths remain unverified. Source: compatibility and research pages. | Defined; not queried live |
| `Q-CUDA13-008` | Performance evaluation · unbranded | Is CPU fallback as fast as GPU execution? | Performance evaluation | No parity claim; transfers and CPU execution can dominate, and broad performance data are unavailable. Source: compatibility and research pages. | Defined; not queried live |

Before any trend or ranking claim, capture the provider, date range, geography, category, query variants, and raw export; do not substitute this intent corpus for those observations.

## Measurement dictionary and current observations

Each dimension is evaluated separately. Empty or unavailable data is `GRAY`, never zero.

| Metric ID | Dimension | Definition and denominator | Source / window | Current state |
|---|---|---|---|---|
| `DPP-AV-01` | Availability | Successful authorized URL fetches / attempted eligible canonical URLs; report response class per URL. | Local HTTP/static checker and public fetch on 2026-11-01. | 15/15 published routes/assets and 9/9 linked project-source URLs returned HTTP 200 for deployed commit `18a075c`; point-in-time availability only. |
| `DPP-ER-01` | Entity recognition | Correct project/author resolution / scored entity opportunities, including ambiguous cases. | Labeled manual query/agent sample, versioned per run. | No observations; `GRAY`. |
| `DPP-RT-01` | Retrieval | Directly traced source retrievals / instrumented retrieval opportunities. A cited or mentioned URL without a trace is an answer mention, not verified retrieval. | Instrumented retrieval logs; provider/model/time required. | No trace access; `GRAY`. |
| `DPP-SF-01` | Semantic fidelity | Supported expected facts preserved / evaluable expected facts in sampled answers. | Human rubric against claim register and versioned query corpus. | No answer sample; `GRAY`. |
| `DPP-EQ-01` | Evidence qualification | Correct support/uncertainty classifications / adjudicated claim rulings. | Human review against claim/source register. | No independent answer rulings; `GRAY`. |
| `DPP-AT-01` | Attribution | Correct source and author attributions / attributable answers. | Manual source/author review of answer samples. | No answer sample; `GRAY`. |
| `DPP-PC-01` | Propagation | Observed eligible surfaces carrying a claim / monitored eligible surfaces. This is monitored coverage, not a census. | Versioned owned/earned/partner surface inventory. | No campaign or monitoring set; `GRAY`. |
| `DPP-FR-01` | Freshness | Time from a recorded correction to first observed corrected representation; uncorrected cases are right-censored. | Dated correction and crawl receipts with sampling interval. | No correction observations; `GRAY`. |
| `DPP-CO-01` | Conversion | Incremental qualified outcomes / eligible randomized or matched exposure units; observed click-through is reported separately. | Requires consented, privacy-reviewed event data and a defensible counterfactual. | No conversion tracking or baseline; `GRAY`. |
| `DPP-BI-01` | Boundary integrity | Passing authorized harmless-fixture trials / conclusive authorized trials. | Separate security-test scope and tool traces; never mixed with visibility scores. | No DPP-ASI boundary experiment; `NOT_APPLICABLE` to current site release. |
| `DPP-SEARCH-01` | Search visibility | Search Console impressions, clicks, indexed URL sample, and query/page dimensions; position is not a universal rank. | Search Console property, extraction date, time window, and eligible-page denominator. | Property not connected; `GRAY`. |
| `DPP-ANALYTICS-01` | Consented site activity | Page views and allowlisted project-link clicks only after affirmative consent; report opt-in coverage and missingness. | GA4, once an owner supplies a property ID and retention configuration. | No measurement ID; no analytics requests or observations. |

The project does not currently have field performance data, search-index coverage, AI answer observations, or a Search Console/GA4 baseline. The consented GA4 implementation remains inactive until a measurement ID is intentionally configured. The local site checker reports implementation properties, not audience outcomes.

## Distribution and intervention ledger

| Activity | Classification | Authority and receipt |
|---|---|---|
| Local website and source changes | Owned, pre-publication | Local work items and `records/evidence.jsonl`; reversible in source control. |
| GitHub repository publication | Published | Public repository `zachshallbetter/cuda-compatibility`; default branch `codex/cuda-compat-publication`; latest Pages deployment was run from `755c6d6cbcea32f886b24fe91f4a48f28fe0bb11` (source assets unchanged from the previous run). Initial ACP mismatch and denied `main` ref requests remain in the history; no `main` ref was created. |
| GitHub Pages site | Live | Workflow run `41311091751` deployed the default branch at `755c6d6cbcea32f886b24fe91f4a48f28fe0bb11`; all 10 sitemap pages, `robots.txt`, `llms.txt`, and both social-card assets returned HTTP 200. This source-only audit update did not change files under `site/`. |
| Social posts, newsletter pitches, community replies, paid promotion | None published | The owner has requested announcements after repository/site publication. No post has been made; destination, method, audience, source asset, and publication receipt remain to be registered per the distribution rule below. |
| Crawl, Search Console, analytics, AI-answer sampling | Not observed or not configured | No Search Console/Bing Webmaster property receipt, GA4 measurement ID, crawler receipt, retrieval trace, or AI-answer sample is recorded. The site has no GA4 requests while the ID is unset; robots and sitemap do not prove processing or indexing. |

Every future distribution action must identify the versioned source asset, destination, method, audience, rights/permission basis, owner, publication receipt, and resulting observations. No automated or mass outreach is part of this project.

## Module coverage and release gates

| DPP module | Implementation level | Evidence and unresolved work |
|---|---|---|
| PRP — representation | Partial, locally implemented | Stable project/site/author IDs, page canonicals, authorship, visible claim/source map, and correction triggers are recorded. Cross-platform identity reconciliation is `GRAY`. |
| DAP — discovery/access | Local controls implemented and public availability observed; search/agent discovery `GRAY` | HTML, internal links, sitemap, robots, and agent index pass local checks; 15 public routes/assets returned 200. Indexation, social rendering, crawler use, and agent retrieval remain unobserved. |
| DIS — distribution | Documented; no campaign | No social, partner, earned, paid, or automated distribution events are recorded. |
| AEP — evidence | Partial, locally implemented | Material technical claims map to primary documentation, implementation records, or named tests. No independent expert review or external corroboration is claimed. |
| OIP — observation | Metrics defined; observations unavailable | Metric definitions, sources, denominators, and blind spots are recorded; no platform baseline is available. |
| OEP — experimentation | Not implemented for presence outcomes | No live SEO, crawler, content, or AI-citation experiment is registered. Site edits and local regression checks are not audience-effect experiments. |
| MPF — measurement/provenance | Partial documentation; not conformant | Seed query corpus and metric dictionary exist; raw external observations, retrieval traces, adjudications, uncertainty estimates, and preregistration receipts do not. |
| ASI — AI surfaces | Not implemented | The site publishes direct HTML and an informal `llms.txt`, and declares selected crawler preferences. It has no instrumented retrieval pipeline or verified answer/citation sample. |
| Release gates | Source and Pages publication verified; external reach and measurement gates pending | `scripts/site_checks.py`, `make check`, and `make test` pass; the exact source commit has a successful Pages run and live routes pass HTTP checks. Search Console/Bing properties, analytics activation, crawler receipts, social-card rendering, audience outcomes, and performance field evidence remain pending or `GRAY`. |

No DPP module is declared conformant. A validator pass, sitemap entry, crawler allowance, answer mention, citation, or local preview cannot be upgraded into an indexing, reach, or causal-outcome claim.

## Maintenance and correction triggers

Review this record when CUDA/PyTorch/tool versions or supported operations change; a material source claim changes; a canonical URL, crawler policy, privacy practice, or analytics configuration changes; the public site is deployed; or new Search Console, analytics, crawler, AI-answer, or distribution evidence is collected. Preserve prior observations and record corrections with their affected claim and representation IDs.
