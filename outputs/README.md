# Numerical outputs

Confirmatory and secondary results (SEED = 20260716, B = 1000):

- `permutation_result.json` — global sign-match test (11/13 observed; null mean 6.3, 95 % interval 2–11; one-sided p = 0.027) and channel-level permutation p-values with BH-FDR flags (6/13 significant)
- `permutation_age.json` — auxiliary age predictions (A1 supplier→escalator, twenties; A2 escalator→landing, thirties onward)
- `prereg_check.json` — automated check of the observed sign matrix against the pre-specified prediction matrix
- `reconfiguration_result.json` — early (2008–2011) vs late (2022–2025) channel-level change, municipality-bootstrap CIs, FDR flags
- `reconfig_trend.json` — annual pipeline intensity 2008–2025, OLS trend (+0.22/yr, p = 0.010), three-period sensitivity split
- `robustness_result.json` — R1 soft allocation, R2 year-matched roles, R3 age subsets
- `effectiveness_all.csv` — pooled directional effectiveness E for all role pairs
- `roles_period_modal.csv` — period-specific modal roles used in RQ2
- `tensor_build_audit.csv` — per-slice flow-conservation audit of the role-flow tensor build (Appendix A)

Large derived tables (`role_flow_long_static.csv`, `roles_annual_modal.csv`) and binary caches (`.npz`) are regenerated deterministically by the pipeline in `code/`.
