# AI answer attribution pilot, baseline results (issue #55)

**Date:** 2026-11-03. **Protocol:** [v0.1.1](PROTOCOL.md). **Raw answers:** [`raw/`](raw/). **Per-answer scores:** [`scores.jsonl`](scores.jsonl) (108 lines, one per product × prompt × run).

Scored against the claim register in [`docs/DPP_IMPLEMENTATION.md`](../../DPP_IMPLEMENTATION.md) and the README "Current status" section. Results are descriptive only.

## Scoring conventions

- **Expected facts** are split into atomic facts from the protocol's "Expected facts" column. Counts used for every run of a prompt: `Q-CUDA13-001` 3 (not officially supported; experimental community approaches exist; selected operations only), `-002` 2, `-003` 3, `-004` 2, `-005` 4, `-006` 2 (registered native operations; bounded CPU fallbacks with limits), `-007` 2, `-008` 2 (no parity; transfers and CPU execution can dominate), `A-001` 4 (experimental; independent; selected PyTorch CUDA 13 operations; `sm_52` Maxwell), `A-002` 1, `A-003` 2 (not an NVIDIA project; independent research), `A-004` 2.
- For unbranded `Q-CUDA13-*` prompts, a correct general answer about stock PyTorch or CUDA 13 is not an unsupported addition; facts about the project are simply not supported.
- **Recognition** is "yes" only when the answer names or describes this project's substance (selected PyTorch CUDA 13 operations on `sm_52`, or its author). Repeating the domain from the prompt is not recognition.
- **NVIDIA misattribution** counts answers scored `wrong:NVIDIA`: the answer names NVIDIA as the creator or owner of the project, or equates the project at `cudacompat.com` with NVIDIA's own CUDA Compatibility framework or `cuda-compat` package.
- A cited `cudacompat.com` link is a mention, not proof the product read the page.
- An answer that was not shown scores 0 facts, recognition "no" for `A-*` and "n/a" for `Q-*`.

## Per-prompt results

Each cell shows runs 1 · 2 · 3 as `supported/expected` facts. `—` = no AI answer shown. `R` = project recognised. `N` = NVIDIA misattribution. `U` = at least one unsupported addition (listed per run in `scores.jsonl`).

| Prompt | Perplexity | Google AI Overview | Bing generated answer |
|---|---|---|---|
| `Q-CUDA13-001` | 1/3 · — · — | 1/3 U · 1/3 · 1/3 U | 0/3 U · 0/3 U · 0/3 U |
| `Q-CUDA13-002` | 2/2 · 2/2 · 2/2 | 2/2 · 2/2 · 2/2 | — · 1/2 U · 1/2 U |
| `Q-CUDA13-003` | 2/3 · — · — | 2/3 · 2/3 · 2/3 | — · 1/3 U · — |
| `Q-CUDA13-004` | 1/2 · — · — | 1/2 U · 1/2 · 1/2 | 1/2 · 1/2 · 1/2 |
| `Q-CUDA13-005` | 0/4 · — · — | 0/4 · 0/4 U · 0/4 U | 0/4 U · 0/4 U · 0/4 U |
| `Q-CUDA13-006` | 0/2 · — · — | 0/2 · 0/2 · 0/2 | 0/2 U · 0/2 U · 0/2 U |
| `Q-CUDA13-007` | 0/2 · — · — | 0/2 U · 0/2 · 0/2 | 0/2 · 0/2 · 0/2 |
| `Q-CUDA13-008` | 2/2 · — · — | 1/2 · 2/2 · 1/2 | 2/2 · 1/2 · — |
| `A-001` | 0/4 · 0/4 · 2/4 R | 0/4 U · 0/4 U · 0/4 U | 0/4 N U · — · 0/4 N U |
| `A-002` | 0/1 · — · — | — · 0/1 N U · 0/1 N U | 0/1 N U · 0/1 N U · 0/1 N U |
| `A-003` | 0/2 · 0/2 · 0/2 | 1/2 U · 1/2 U · 1/2 U | — · — · — |
| `A-004` | — · — · — | 0/2 · 0/2 N U · 0/2 N U | 0/2 U · 0/2 U · 0/2 U |

## Totals per product

| Product | Answers shown (of 36) | Project recognised | Correct author | NVIDIA misattribution | Answers with unsupported additions | `cudacompat.com` cited |
|---|---|---|---|---|---|---|
| Perplexity | 17 | 1 | 0 | 0 | 0 | 1 |
| Google AI Overview | 35 | 0 | 0 | 4 | 16 | 0 |
| Bing generated answer | 28 | 0 | 0 | 5 | 20 | 0 |

No answer on any product named Zach Shallbetter. The one recognition (Perplexity `A-001` run 3) cited `https://www.cudacompat.com/` and said it used the site's search-indexed description because the page itself failed to load.

## `cudacompat.com` in first-page results

| Product | Runs where `cudacompat.com` appeared in first-page results |
|---|---|
| Google | 0 of 36 |
| Bing | 0 of 36 |

On Google, the project appeared in first-page results in 14 of 36 runs only through the retired GitHub Pages URL `zachshallbetter.github.io/cuda-compatibility` (now 404): `Q-CUDA13-001` run 3, all runs of `-002`, `-003`, `-005` and `-006`, and `-007` run 3. On Bing, the only `cudacompat.com` text matches were `bing.com/search` links echoing the query.

## Notable misattributions (verbatim)

1. Bing, `A-002` run 1: "The CUDA Compatibility project is officially developed and maintained by NVIDIA, not by an independent individual or third‑party group."
2. Bing, `A-002` run 3: "The cudacompat.com site is not an official NVIDIA domain; it is a third‑party repository that hosts the CUDA Compatibility Package"
3. Bing, `A-001` run 1: "CUDA Compatibility at cudacompat.com (and NVIDIA’s docs) explains the rules, models, and packages"
4. Google, `A-002` run 2: "The CUDA Compatibility platform and its official tools are created and maintained by NVIDIA."
5. Google, `A-004` run 2: "The term cudacompat (or cuda-compat) does not refer to an independent community hardware project or website"
6. Google, `A-004` run 3: "There is no standalone project, website, or entity known as \"cudacompat.com\"."

## Limitations

- **Browser state.** All collection used one persistent browser profile, signed out. It carried cookies, so "signed out" doesn't mean no history or location signals.
- **Perplexity refusals.** Perplexity refused 19 of 36 runs with a sign-up request when signed out (all of `A-004`, and runs 2–3 of every other prompt except `Q-CUDA13-002`, `A-001` and `A-003`). From about 13:13 PST it also hid the source panel behind a sign-up wall, so most Perplexity source lists hold only the inline citations shown, and "+1" chips were not expanded.
- **Bing read timing.** Bing answers were read a fixed number of seconds after page load (about 9–18 s). Answers that render later may have been missed; 8 Bing runs recorded no generated answer (5 showed a featured snippet instead, 3 showed neither).
- **Google sources are display labels.** Google AI Overview sources were captured as the labels shown (site names, link text, even product names such as "RTX 5090"), not URLs. `cudacompat_cited` for Google is therefore based on labels only.
- **Method tests.** `A-002` run 1 on each product was a method-selection test run before bulk collection. Google's showed no AI Overview; Bing's text was truncated at 800 characters with no sources captured.
- **Collection order and window.** All 108 answers were collected on one day between about 13:08 and 13:35 PST, with the three products collected in overlapping windows rather than one product at a time.
- **Dependent runs.** The three runs of one prompt on one product are not independent samples and are not pooled into rates.
- **Scoring.** One scorer, judging against the project's own claim register; the register is self-authored project evidence. Fact counts depend on the atomic split stated above.
- **Descriptive only.** These results record what the products said on this date. They make no causal claim about why any product answered as it did.
