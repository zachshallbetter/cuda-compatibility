# Launch and distribution workflow

This folder prepares publication around one reproducible result: the [public `sm_52` example](../../examples/sm52-mlp/README.md) and the evidence behind it. It holds drafts and plans only. **Posting, outreach and spending each need the owner's specific approval; nothing here has been posted.**

| File | Contents |
|---|---|
| [article.md](article.md) | Implementation article: runtime facade, native kernels, bounded fallback, limits |
| [social.md](social.md) | Reveal short, limitation short, LinkedIn post, X thread; why HN text is not drafted |
| [channels.md](channels.md) | Destination shortlist with rules, fit, class, rights, cost, owner and readiness |

The placement ledger is the **Distribution and intervention ledger** in [`docs/DPP_IMPLEMENTATION.md`](../DPP_IMPLEMENTATION.md#distribution-and-intervention-ledger), not a second register.

## Rules

1. **One source of facts.** Every draft reuses the article's claims and evidence IDs. A repost, quote or share of the same claim is not new evidence. Only a new run, ideally by someone else, adds evidence.
2. **Exact limits travel with the claim.** Don't cut a limit to fit a format. If a post can't carry it, link to where it is stated.
3. **One next action per draft:** run the example, or file a [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml). Every draft links the canonical source, https://cudacompat.com/ or the example.
4. **No generated text where a destination forbids it.** Hacker News is one. The owner writes those posts.
5. **No automated or mass outreach**, and no vote, like or comment solicitation.

## Draft and placement states

Drafts and placements are tracked separately. A draft can be ready while no placement exists.

| Draft state | Meaning |
|---|---|
| Draft | Written here, facts checked against the evidence at the commit that changed it |
| Owner-edited | The owner has rewritten it in their own voice; the facts are re-checked |
| Approved | The owner has approved one specific destination and date |

| Placement state | Meaning |
|---|---|
| Planned | Destination, date and draft version chosen; rules re-read that day |
| Posted | URL, timestamp and the exact text posted are recorded in the ledger |
| Removed or corrected | What changed and why, with the earlier receipt kept |

Current state: all drafts are **Draft**; no placements exist.

## Reviews after a post

Run each review at D+7, D+28 and D+90 after the first placement. Keep the four outcomes separate and report each one as a count with its source, or as `GRAY` when there is no data:

| Outcome | Measure | Source | Not the same as |
|---|---|---|---|
| Access | Sessions, example page views, source downloads, command copies (visitors who have not opted out); search impressions and clicks | GA4 (`DPP-ANALYTICS-01`), Search Console (`DPP-SEARCH-01`) | Interest or use; opt-outs make coverage partial |
| Recognition | Correct project and author identification in sampled answers or third-party mentions | Manual sample (`DPP-ER-01`, `DPP-AT-01`) | Traffic |
| Independent reproduction | Reproduction reports from people other than the author, split by outcome (matched, differed, could not run) | GitHub issues using the reproduction form | Stars, likes or comments |
| Qualified inquiries | Issues or messages asking about a specific card, workload or contribution | GitHub issues and the owner's inbox, classified by hand | Generic praise or vote counts |

Platform engagement numbers (likes, reposts, upvotes) may be recorded as observed, but they don't count toward any of the four outcomes. Nothing is extrapolated from one window to another.

## Costs and missing data

- **Money:** none planned. Paid placement is out of scope.
- **Time:** editing drafts, recording two short videos, and staying available to answer questions on the day of each post (required for Show HN).
- **Missing data:** no audience size, follower count, referral history or reach estimate exists for any destination. None is assumed. Sponsorship value is not estimated.

## How testers report results

Testers follow [`examples/sm52-mlp/README.md`](../../examples/sm52-mlp/README.md) and open a [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml). The form asks for the project revision, GPU and driver, software versions, `doctor` output, both runs' JSON and exit codes, and whether the result matched. A report from separate hardware is recorded as independent evidence only after its output has been read and classified.
