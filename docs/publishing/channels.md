# Channel shortlist

**Checked:** 2026-11-03. Rules change, so re-read each destination's rules on the day you post and record the check in the ledger. "Not observed" means the page was not read, not that the destination has no rule.

**Audience data:** none. The project has no follower counts, referral history or reach measurements for any destination, and none are estimated here. Fit is a judgment from each destination's stated topic.

| Destination | Class | Audience fit | Rules observed (source, date) | Format | Attribution and rights | Cost | Owner | Readiness |
|---|---|---|---|---|---|---|---|---|
| cudacompat.com article page | Owned | Anyone arriving from the posts below | Project's own | [Article draft](article.md) | Author byline; MIT/Apache project text | None | Owner | Ready once the owner edits the draft |
| GitHub release notes / Discussions | Owned | Existing repository visitors | Project's own | Short summary plus link | Same | None | Owner | Ready |
| LinkedIn (owner profile) | Owned profile, algorithmic reach | Professional network; reach unknown | Not observed | [LinkedIn draft](social.md#linkedin) | Owner's account | None | Owner | Ready once the owner edits the draft |
| X (owner profile) | Owned profile, algorithmic reach | GPU/ML practitioners who follow the owner; reach unknown | Not observed | [Thread draft](social.md#x-thread) | Owner's account | None | Owner | Ready once the owner edits the draft |
| YouTube Shorts or similar | Owned channel, algorithmic reach | Unknown | Not observed | [Two short scripts](social.md) | Owner-recorded footage of the owner's machine; no NVIDIA assets beyond public product names | Recording time | Owner | Needs recording |
| Hacker News, Show HN | Earned | General technical readers | [Show HN](https://news.ycombinator.com/showhn.html) and [guidelines](https://news.ycombinator.com/newsguidelines.html), 2026-11-03: must be something people can try, ideally without barriers; you must have worked on it and be present to discuss it; no generated text; no vote solicitation; no landing pages | Owner-written title starting "Show HN:" plus a first comment | Link the repository or example, not a landing page | None | Owner | **Hold.** It is installable, but trying it needs an `sm_52` GPU, LLVM 23.1.4 and CUDA 12.4 headers. Recommended gate: at least one outside reproduction report first, so the post can say it works beyond the author's machine |
| Lobsters | Earned | Systems and compiler readers | [About](https://lobste.rs/about), 2026-11-03: invite-only; self-promotion under about a quarter of your activity; new users can't submit self-promotion for 70 days; tags required | Link post | Same | None | Owner | **Not eligible** unless the owner has an established account |
| PyTorch Forums | Earned / community | PyTorch users with legacy GPUs | [Guidelines](https://discuss.pytorch.org/guidelines), 2026-11-03: right category, no cross-posting, no spam; no stated self-promotion rule | One post in the most relevant category, or replies to existing threads about Maxwell and CUDA 13 where it answers the question | Same | None | Owner | Ready; reply only where it is on-topic |
| r/CUDA, r/LocalLLaMA, r/MachineLearning | Earned / community | Practitioners with older cards | **Not observed**: Reddit blocked automated reads on 2026-11-03 | Text post linking the example | Same | None | Owner | Check each subreddit's rules (self-promotion ratio, required flair, project-showcase threads) by hand first |
| Specialist publications and newsletters (e.g. GPU or ML engineering newsletters) | Earned (pitch) or partner | Unknown per publication | Not observed | Short pitch linking the article and evidence | Publication's terms | Unknown; reject any paid placement unless separately authorized | Owner | Not started; list specific publications only after reading their submission terms |
| Paid placement | Paid | — | — | — | — | Not budgeted | Owner | **Out of scope.** Needs separate authority and budget |

Upstream engineering contributions (issue #44) are a separate track and not publicity.
