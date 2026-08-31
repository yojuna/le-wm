# lewm_phi_v2_eval run notes

- Date: 2026-08-28
- Does not modify `eval_results/pusht/lewm_phi_v1/`
- Protocol: online_offset, short_horizon, n=20, seed=0, collect-episodes=320
- Weights:
  - E1: l2_z (no φ)
  - E2_phi_fixed: stablewm/checkpoints/pusht/lewm_phi_fixed/reach.pt
  - E2_phi_v2: stablewm/checkpoints/pusht/lewm_phi_v2/reach.pt
  - E4: random φ
- Summary: ../../../../docs/lewm_phi_v2_summary.md
