# From regional roles to migration flows

Replication repository for:

> Kim, D. *From regional roles to migration flows: Testing the life-course division of labour in South Korea, 2006–2025.* (submitted)

The study tests whether the six life-course functional roles measured in the companion study — Kim, D. (2026). *From escalator region to escalator system: The life-course division of labor in South Korea's internal migration, 2006–2025* [Preprint]. Research Square ([repository](https://github.com/dongwoo2022008/From-escalator-region-to-escalator-system)) — organise the direction of the complete origin–destination migration network of South Korea's 229 municipalities (93,950,706 inter-municipal moves, 2006–2025), against 13 pre-specified directional predictions.

## Structure

| Directory | Contents |
|---|---|
| `prespecification/` | Dated pre-specification document (predicted-sign matrix, pair partition, success criterion ≥ 11/13, auxiliary age predictions A1–A3, decision rules) — fixed before any role-to-role flow had been computed. Korean registered original (verbatim) plus an English translation |
| `code/` | Full analysis pipeline and figure scripts (Python 3; numpy, matplotlib) |
| `data/` | Harmonisation crosswalk (`sgg_harmonize_map.csv`) and inherited role assignments from the companion study (fixed inputs; not re-estimated here) — see `data/README.md` |
| `outputs/` | Numerical results: permutation inference, reconfiguration, robustness, effectiveness matrices, role tables, and the tensor build audit (flow-conservation checks of Appendix A) |

## Reproduction order

All analyses run as a single scripted workflow with a fixed random seed (`SEED = 20260716`):

```
code/build_role_flow_tensor.py      # OD register → 6×6×8×20 role-flow tensor (+ build audit)
code/compute_effectiveness.py       # demographic effectiveness E_ij
code/permutation_test.py            # confirmatory test (B = 1000 label permutations, BH-FDR)
code/permutation_age.py             # auxiliary age predictions
code/reconfiguration.py             # RQ2 two-window comparison
code/reconfig_trend.py              # RQ2 annual intensity trend (municipality bootstrap)
code/robustness.py                  # R1 soft allocation · R2 year-matched roles · R3 age subsets
code/make_figures.py  code/make_figure1.py  code/make_figure0b.py
code/make_figure5_combined.py  code/make_figure5.py  code/make_figure_s1.py
```

Intermediate binary caches (`.npz`), manuscript figure files, and the large derived tables (`role_flow_long_static.csv`, `roles_annual_modal.csv`, calibrated `memberships.csv`) are not tracked; all are regenerated deterministically by the pipeline above (fixed seed), and the membership probabilities are produced by the companion study's measurement pipeline.

## Data availability

The underlying municipality-level records of the Internal Migration Statistics are supplied by Statistics Korea under restricted-access conditions and cannot be redistributed; aggregated origin–destination flows are publicly available from Statistics Korea (KOSIS, https://kosis.kr). The header of `code/build_role_flow_tensor.py` documents the exact extraction and processing path (raw file layout, harmonisation, exclusion rules) so that qualified researchers can regenerate the analytical dataset from the original source and reproduce the reported analyses.

## Author

Dongwoo Kim — Division of Advanced IT, Baekseok University (dongwoo.kim@bu.ac.kr, ORCID 0000-0003-2219-083X)
