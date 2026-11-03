# AI answer attribution pilot, baseline (issue #55)

**Version:** 0.1.1, fixed before bulk collection on 2026-11-03. **Locale:** en-US (`hl=en&gl=us` / `setlang=en-us&cc=us` where the product accepts it). **Collector:** Claude, with the owner's approval, in the Claude desktop app's built-in browser. **Scorer:** a separate pass against the claim register; scoring rules below.

This pilot records how public AI answer products describe the project, its author and its support limits shortly after `cudacompat.com` launched. It is a dated baseline for later comparison, not a measurement of any effect. Expect many answers not to know the project yet; that is a valid result.

## Products

| ID | Product | How | Session state observed |
|---|---|---|---|
| `perplexity` | Perplexity | `perplexity.ai/search?q=<prompt>`; answer plus every cited source | Signed out ("Sign in to save your history") |
| `google` | Google Search AI Overview | `google.com/search?q=<prompt>&hl=en&gl=us`; the AI Overview if one is shown, plus whether `cudacompat.com` appears in the first page of results | Signed out |
| `bing` | Bing Search with its Copilot answer | `bing.com/search?q=<prompt>&setlang=en-us&cc=us`; the generated answer if shown, plus whether `cudacompat.com` appears in the first page of results | Signed out |

**Changed from 0.1.0 before bulk collection.** ChatGPT and the Copilot chat app were dropped. The built-in browser was signed in to the owner's ChatGPT workspace, which offered paid credits and whose temporary chat can still use memory and custom instructions. Copilot chat automatically signed in to the owner's work Microsoft 365 tenant. Neither was a clean or free session, and signing the owner out was out of scope. Bing's search answer is a separate, signed-out Microsoft surface. One test query per product (`A-002`) was run while choosing the method; those test answers are kept as run 1 of `A-002`.

**Session note.** The browser has a persistent profile with cookies, so "signed out" doesn't mean no history or location signals.

## Prompts

Type each prompt exactly. Run it **3 times per product, each in a fresh chat** (new temporary chat or new private window). If a product refuses or gives no AI answer, record that; don't rephrase.

| Prompt ID | Family | Prompt | Expected facts (for scoring) |
|---|---|---|---|
| `Q-CUDA13-001` | Feasibility | Can I run CUDA 13 PyTorch on an older NVIDIA GPU? | Not officially; experimental community approaches exist; selected operations only |
| `Q-CUDA13-002` | Toolkit policy | Does CUDA 13 support Maxwell? | CUDA 13 dropped Maxwell, Pascal and Volta from the toolkit; that is separate from what the hardware can compute |
| `Q-CUDA13-003` | Card lookup | Can a GTX 980 Ti run a CUDA 13 PyTorch wheel? | GTX 980 Ti is `sm_52`; the official wheel has no kernels for it; any project support is unverified for that card |
| `Q-CUDA13-004` | Error diagnosis | What does "no kernel image is available" mean in PyTorch? | The binary has no image for the GPU's architecture; version spoofing can't supply kernels |
| `Q-CUDA13-005` | Setup | How do I install a CUDA compatibility layer for an old GPU? | Same Python environment as PyTorch; Linux x86_64; user-installed LLVM and CUDA headers; opt-in launcher |
| `Q-CUDA13-006` | Support lookup | Which PyTorch CUDA 13 operations work on sm_52? | Only registered native operations (add, arange, embedding, layer norm, matmul, ReLU) and bounded CPU fallbacks, with limits |
| `Q-CUDA13-007` | Training boundary | Can I train or use autograd on Maxwell with this tool? | Only a tiny first-order SGD case is shown; broad training is unverified |
| `Q-CUDA13-008` | Performance | Is CPU fallback as fast as GPU execution? | No parity claim; transfers and CPU execution can dominate |
| `A-001` | Recognition | What is CUDA Compatibility at cudacompat.com? | An experimental, independent project running selected PyTorch CUDA 13 operations on `sm_52` Maxwell GPUs |
| `A-002` | Author attribution | Who created the CUDA Compatibility project at cudacompat.com? | Zach Shallbetter |
| `A-003` | Affiliation | Is cudacompat.com an official NVIDIA project? | No; independent research, not affiliated with or endorsed by NVIDIA |
| `A-004` | Support scope | Which GPUs has the cudacompat.com project actually tested? | GTX TITAN X (GM200, `sm_52`) reference cards only; other cards are unverified candidates |

12 prompts × 3 runs × 3 products = 108 answers. Run all of one product before moving on, and note the start and end times.

## What to record

For each answer, paste it into that product's file in this folder (template: [`TEMPLATE.md`](TEMPLATE.md)):

- prompt ID and run number (1–3)
- time (local, with time zone)
- model or mode name, if the product shows one; otherwise "not shown"
- the **full answer text**, unedited
- every cited link or source, in order, exactly as shown
- whether you were logged out or in a temporary chat

Don't summarize, fix or trim answers. Leave missing fields blank; blank means not observed.

## Scoring (done afterwards, not during collection)

Each answer gets four separate scores against the claim register in `docs/DPP_IMPLEMENTATION.md`:

1. **Recognition:** names or clearly describes this project (yes / no / not applicable for unbranded prompts that don't need it).
2. **Attribution:** names the correct author; names a wrong author; or names none.
3. **Factual support:** the share of the prompt's expected facts that are stated correctly.
4. **Unsupported additions:** any claim the evidence doesn't support, such as broad CUDA 13 support, NVIDIA endorsement, untested cards described as working, or performance parity.

A cited `cudacompat.com` link is a **mention**, not proof the product read the page. The three runs of one prompt on one product are dependent, so results are reported per prompt and product, with all three runs shown, not pooled as independent samples. Findings are descriptive only.
