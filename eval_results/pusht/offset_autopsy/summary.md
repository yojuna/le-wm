# Offset autopsy Phase A — results

**Seed:** 0 · **pairs:** 24 · **offset:** 25
**φ weights:** `/home/aj/code/ssd_repos/robo_ops/ws_weltmodelle/ws_le-wm/stablewm/checkpoints/pusht/lewm_phi_v2/reach.pt`

## Hypothesis call

**Primary:** `H1_imagination_ood`

| Flag | Value |
|------|-------|
| H1_imagination_ood | True |
| H2_collapsed_spread | True |
| H3_weak_real_ranking | False |
| H4_all_costs_weak | False |

### Notes

```json
{
  "phi_spearman": 0.9911396011396012,
  "l2_spearman": 0.9961538461538462,
  "random_spearman": 0.9945299145299146,
  "phi_rel_gap": 10471789.618333181,
  "l2_rel_gap": 11594232.360521952,
  "phi_cost_std": 0.4809139172236125,
  "l2_cost_std": 6.313459078470866
}
```

## Real-path ranking (cost vs remaining steps)

| Cost | Spearman (mean±std) | frac decreasing |
|------|---------------------|-----------------|
| l2_z | 0.996±0.005 | 0.907 |
| phi | 0.991±0.014 | 0.847 |
| random | 0.995±0.007 | 0.868 |

## Imagination gap (end of path, true actions)

Mean ‖ẑ−z‖₂ along path: **6.311**

| Cost | relative end-gap mean±std |
|------|---------------------------|
| l2_z | 11594232.361±4002380.347 |
| phi | 10471789.618±3515657.394 |
| random | 6510510.455±1975544.587 |

## Candidate cost spread

| Cost | mean | std | cv |
|------|------|-----|----|
| l2_z | 123.471 | 6.313 | 0.061 |
| phi | 8.364 | 0.481 | 0.066 |
| random | 6.588 | 0.213 | 0.033 |

## Artifacts

- `/media/aj/dirk/aj/code/repos/robo_ops/ws_weltmodelle/ws_le-wm/le-wm/eval_results/pusht/offset_autopsy/real_progress.png`
- `/media/aj/dirk/aj/code/repos/robo_ops/ws_weltmodelle/ws_le-wm/le-wm/eval_results/pusht/offset_autopsy/imagination_gap.png`
- `/media/aj/dirk/aj/code/repos/robo_ops/ws_weltmodelle/ws_le-wm/le-wm/eval_results/pusht/offset_autopsy/cost_spread.png`
- `/media/aj/dirk/aj/code/repos/robo_ops/ws_weltmodelle/ws_le-wm/le-wm/eval_results/pusht/offset_autopsy/summary.json`
