# Paper IV — Pre-registration of predicted directional signs (English translation)

> [English translation of the registered Korean original (`preregistration_predicted_signs_v1_korean_original.md`), provided for reference. In case of any discrepancy, the Korean original governs.]

> **Registration-timing principle:** This table is fixed **before** any role-to-role directional flow (E_ij) is observed.
> Up to the point of registration, the only quantities inspected were (i) the harmonisation integrity of the OD files (160 files, zero unmapped records), (ii) self-loop shares, and (iii) the existence of regions;
> **no directionality or net exchange between role pairs was observed in any form.** → defence against the "predictions were fitted after the fact" objection (brief §9).
> Registered: 2026-07-21 · SEED = 20260716 · The measurement instrument (the roles) was fixed once in the companion study (Study III); no re-derivation in Study IV.

---

## 1. Test statistic

For each role pair (i, j), **demographic effectiveness**
E_ij = 100 · N_ij / T_ij,  N_ij = F_{i→j} − F_{j→i},  T_ij = F_{i→j} + F_{j→i}
(De Jong et al. 2016; Plane & Jurjevich 2009). Range −100…+100. Antisymmetry: E_ij = −E_ji.
**E_ij > 0 ⇔ role i sends migrants net to role j (i is a net supplier relative to j).**

Six roles (inherited from Study III, fixed): supplier–return (SUP) · low-mobility anchor (ANC) · gradual outflow (OUT) · escalator (ESC) · landing zone (LAND) · high-turnover reception (HTR).

---

## 2. Theoretical pipeline (division-of-labour hypothesis → grounds for the directional predictions)

Core claim of brief §2: the division of labour is not a mere coexistence of roles but a **circulation system interlocked by flows**, and the predicted pipeline is

**[net sources: anchor · gradual outflow] · [supplier–return] → [escalator] → [landing zone / high-turnover reception]**

formalised as life-course positions (pipeline levels; net migration flows from low to high):

| level | roles | life-course function |
|---|---|---|
| 0 (deep sources) | low-mobility anchor (ANC), gradual outflow (OUT) | net youth outflow, low mobility; the system's primary supply |
| 1 (supplier–return hub) | supplier–return (SUP) | supplies the young to escalators + receives returns |
| 2 (escalator) | escalator (ESC) | accumulates the young, releases them at family-formation ages |
| 3 (landing / reception) | landing zone (LAND), high-turnover reception (HTR) | receives the released settlement and inflow (terminal reception) |

Prediction rule: **level(i) < level(j) ⇒ E_ij > 0** (net migration from lower to higher level).
Same-level pairs (ANC↔OUT, LAND↔HTR): **no directional prediction (excluded from the confirmatory test)**.

---

## 3. Pre-registered sign matrix (row i → column j; sign = predicted E_ij)

`+` = predicted net outflow i→j (E_ij>0) · `−` = predicted net inflow (E_ij<0) · `.` = diagonal (self, excluded by definition) · `?` = no directional prediction (same level)

|  i \\ j | SUP | ANC | OUT | ESC | LAND | HTR |
|---|---|---|---|---|---|---|
| **SUP**  (L1) | . | − | − | + | + | + |
| **ANC**  (L0) | + | . | ? | + | + | + |
| **OUT**  (L0) | + | ? | . | + | + | + |
| **ESC**  (L2) | − | − | − | . | + | + |
| **LAND** (L3) | − | − | − | − | . | ? |
| **HTR**  (L3) | − | − | − | − | ? | . |

---

## 4. Confirmatory core predictions (falsifiable; the cells on which the paper's claims rest)

Numbered predictions with sign and rationale. **P1–P3 are the heart of the pipeline** (the strongest falsifiable predictions).

- **P1. SUP → ESC : +** — supplier–return supplies the young to escalators (pipeline stage 1).
- **P2. ESC → LAND : +** — escalators release family-formation-age population to landing zones (stage 2).
- **P3. ESC → HTR : +** — the other terminal of escalator release (stage 2).
- **P4. ANC → {ESC, LAND, HTR} : +** — net supply from the low-mobility anchor (the system's primary source).
- **P5. OUT → {ESC, LAND, HTR} : +** — net supply from gradual outflow.
- **P6. SUP → {LAND, HTR} : +** — supplier–return is also a net supplier to the reception roles.
- **P7. ANC → SUP : + , OUT → SUP : +** — deep sources supply the supply hub net (level 0→1).
- **P8 (reception absorption).** LAND and HTR must be net receivers from every other role (all `−` in the LAND and HTR rows of the matrix above).

**Confirmatory target = the signs of the 13 unordered pairs** (= 13 positive directed cells plus their 13 negative reverses).
**No directional prediction (2 pairs): ANC↔OUT, LAND↔HTR** — excluded from sign scoring (magnitude and significance reported only).

## 5. Auxiliary age-consistency predictions (auxiliary to the RQs; tested on the 8-age-group tensor)

- **A1.** The positive intensity of SUP→ESC (P1) is **greatest in the twenties** and declines towards older ages.
- **A2.** ESC→LAND (P2) **emerges and strengthens in the thirties** (family-formation release) and is comparatively weak in the twenties.
- **A3.** Net inflow into LAND and HTR (P8) is most pronounced in the thirties–forties.
- **A4.** At older ages (sixties, seventies and over), the directional pipeline may weaken or reverse into a return (homecoming) component — **no sign prediction (exploratory)**.

## 6. Decision rules (applied after observation)

1. **Sign-match rate:** the share of the 13 confirmatory unordered pairs whose observed E_ij sign matches the prediction. Pre-specified success criterion: **≥ 11/13 (85%)**.
2. **Significance:** permutation (B = 1000, SEED = 20260716) with BH-FDR at 5% for |E_ij| (or net flow). Report the number of cells that are in the predicted direction and FDR-significant.
3. **Age consistency:** whether the directions of A1–A3 hold in the corresponding age slices (sign + relative intensity).
4. RQ2 (reconfiguration) is judged separately as strengthening/weakening of the above structure in the early (2008–2011) and late (2022–2025) windows.

> The decision rules and success criterion were fixed before observation. In all subsequent stages this file is used **only as an unmodified reference standard**.
