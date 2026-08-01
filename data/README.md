# Fixed inputs

- `sgg_harmonize_map.csv` — raw municipality code → harmonised 229-municipality panel (blank panel code = dropped; the pre-2012 Sejong predecessor county). Identical to the crosswalk of the companion study.
- `role_assignment_229.csv` — modal life-course functional role of each municipality (window-mean argmax), inherited unchanged from the companion study (Kim 2026): https://github.com/dongwoo2022008/From-escalator-region-to-escalator-system
- Calibrated membership probabilities (`memberships.csv`, used by robustness check R1) are produced by the companion study's measurement pipeline and are regenerated from it; they are also included in the journal submission package.

Raw origin–destination records are supplied by Statistics Korea under restricted-access conditions and are not redistributed here; see the repository README.
