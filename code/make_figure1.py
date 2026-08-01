# -*- coding: utf-8 -*-
"""Manuscript Figure 1: confirmatory study design (method flow).
v4.1 2026-08-01 — monochrome journal style; arrows now purely vertical (axis-aligned)
per user direction. Boxes white with black edges, no colour anywhere.
Overwrites F0_method_flow.png. Previous colour version kept as F0_method_flow_OLD_v3_colour.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

EDGE = "#1a1a1a"
FACE = "#ffffff"
ARROW = "#1a1a1a"
TAG = "#777777"

fig, ax = plt.subplots(figsize=(13.5, 9.2), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(cx, cy, w, h, tag, title, body, lw=1.6, tfs=14.5, bfs=11.8):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.5,rounding_size=0.8",
                 linewidth=lw, edgecolor=EDGE, facecolor=FACE, zorder=3))
    ax.text(cx + w/2 - 1.5, cy + h/2 - 1.2, tag, ha="right", va="top",
            fontsize=10.5, color=TAG, style="italic", zorder=4)
    ax.text(cx, cy + h*0.22, title, ha="center", va="center",
            fontsize=tfs, color="#000000", fontweight="bold", zorder=4)
    ax.text(cx, cy - h*0.20, body, ha="center", va="center",
            fontsize=bfs, color="#333333", zorder=4, linespacing=1.5)

# boxes (geometry identical to v3)
box(50, 88, 74, 15, "§3.1", "Inherited role classification  (fixed, not re-estimated)",
    "classification of 229 municipalities into six roles\nGaussian mixture model on marginal age profiles only")
box(27, 61, 42, 19, "§3.3", "Theoretical predictions\n(fixed before observation)",
    "four-level pipeline hierarchy\nlevel(i) < level(j)  ⇒  E$_{ij}$ > 0\n13 signed pairs · criterion ≥ 11/13",
    tfs=14, bfs=11.5)
box(73, 61, 42, 19, "§3.2 · §3.4–3.5", "OD migration register",
    "93.95 M inter-municipal moves, 2006–2025\nharmonised to the 229-municipality panel\naggregated to role-to-role flows",
    tfs=14, bfs=11.5)
box(50, 33, 68, 15, "§3.6", "Confirmatory evaluation  (RQ1)",
    "observed vs pre-specified directional signs\npermutation null: role labels reassigned · channel-level FDR",
    lw=2.2)
box(27, 9.5, 42, 13, "§3.7", "Reconfiguration  (RQ2)",
    "early vs late coupling · annual intensity trend", tfs=13.5, bfs=11.5)
box(73, 9.5, 42, 13, "§3.8", "Robustness",
    "membership-weighted · year-matched · age subsets", tfs=13.5, bfs=11.5)

def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                 linewidth=1.4, color=ARROW, zorder=2, shrinkA=0, shrinkB=0,
                 connectionstyle="arc3,rad=0"))

# flow: inheritance -> predictions & data; both -> evaluation; evaluation -> secondary
# purely vertical arrows at the column centres (x = 27 and 73)
arrow((27, 80.0), (27, 71.3))
arrow((73, 80.0), (73, 71.3))
arrow((27, 50.8), (27, 41.3))
arrow((73, 50.8), (73, 41.3))
arrow((27, 24.8), (27, 16.8))
arrow((73, 24.8), (73, 16.8))

fig.savefig("/home/claude/paper4_work/figures/F0_method_flow.png",
            bbox_inches="tight", facecolor="white")
print("saved")
