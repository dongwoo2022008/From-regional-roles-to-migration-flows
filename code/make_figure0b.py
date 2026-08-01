# -*- coding: utf-8 -*-
"""Manuscript Figure 2: hypothesised pipeline hierarchy (journal-restrained style).
v5.1 2026-08-01 — monochrome; all arrows straight and vertical (axis-aligned) per user
direction: principal channels as vertical black arrows, other lower-to-higher channels as
straight vertical grey arrows at the outer margins (no curves).
Previous colour version kept as F0b_pipeline_hierarchy_OLD_v4_colour.png. Levels 1-4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

EDGE = "#1a1a1a"  # monochrome: single edge colour for all role boxes
GREEN = ORANGE = RED = BLUE = EDGE
GREY = "#9a9a9a"
LGREY = "#c8c8c8"
DARK = "#2b2b2b"

fig, ax = plt.subplots(figsize=(14.2, 9.2), dpi=150)
ax.set_xlim(0, 145); ax.set_ylim(0, 100); ax.axis("off")

def box(cx, cy, w, h, text, color, fs=16):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.4,rounding_size=1.2",
                 linewidth=2.0, edgecolor=color, facecolor="white", zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#222222", zorder=4)

# role boxes (bottom -> top)
box(33, 13, 24, 9, "low-mobility\nanchor", GREEN)
box(63, 13, 24, 9, "gradual\noutflow", GREEN)
box(48, 37, 30, 9, "supplier–return", ORANGE)
box(48, 60, 30, 9, "escalator", RED)
box(33, 84, 24, 9, "landing zone", BLUE)
box(63, 84, 24, 9, "high-turnover\nreception", BLUE)

# dashed same-level links
ax.plot([45.5, 50.5], [13, 13], ls=(0, (4, 3)), color=GREY, lw=2.0, zorder=2)
ax.plot([45.5, 50.5], [84, 84], ls=(0, (4, 3)), color=GREY, lw=2.0, zorder=2)

def arrow(p0, p1, color=DARK, lw=2.8, rad=0.0, ms=20, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                 linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=z,
                 shrinkA=0, shrinkB=0))

# principal feed and release pathways (solid black, vertical)
arrow((40, 18.0), (40, 32.0))
arrow((56, 18.0), (56, 32.0))
arrow((48, 42.0), (48, 55.0))
arrow((40, 65.0), (40, 79.0))
arrow((56, 65.0), (56, 79.0))

# remaining lower->higher pairs (straight vertical grey, outer margins)
arrow((25, 18.0), (25, 79.0), color=GREY, lw=1.6, ms=14)
arrow((71, 18.0), (71, 79.0), color=GREY, lw=1.6, ms=14)

# dotted level separators (behind everything)
for y in (25, 48.5, 72):
    ax.plot([4, 78], [y, y], ls=(0, (1.5, 2.5)), color=LGREY, lw=1.3, zorder=1)

# left bracket + horizontal level labels
ax.plot([4, 4], [5, 92], color=LGREY, lw=1.4, zorder=1)
for y in (5, 25, 48.5, 72, 92):
    ax.plot([4, 5.4], [y, y], color=LGREY, lw=1.4, zorder=1)
ax.text(4, 96.5, "Level", ha="left", va="center", fontsize=16,
        color=DARK, fontweight="bold")
for num, sub, y, c in [("4", "Sinks", 84, BLUE),
                       ("3", "Escalator", 60, RED),
                       ("2", "Supply stage", 37, ORANGE),
                       ("1", "Deep sources", 13, GREEN)]:
    ax.text(9.5, y + 2.2, num, ha="center", va="center", fontsize=21,
            color=DARK, fontweight="bold")
    ax.text(9.5, y - 3.4, sub, ha="center", va="center", fontsize=11.5, color="#555555")

# compact legend (right)
ax.add_patch(FancyBboxPatch((84, 28), 56, 48,
             boxstyle="round,pad=1.2,rounding_size=1.6",
             linewidth=1.1, edgecolor="#cccccc", facecolor="white", zorder=2))
ax.text(87, 71, "Prediction rule", ha="left", va="center", fontsize=13.5,
        color=DARK, fontweight="bold", zorder=4)
ax.text(87, 65.5, "level(i) < level(j)  ⇒  E$_{ij}$ > 0", ha="left", va="center",
        fontsize=13, color="#333333", zorder=4)
arrow((87, 56), (95, 56), color=DARK, lw=2.8, ms=18, z=4)
ax.text(98, 56, "Principal feed and release\nchannels", ha="left", va="center",
        fontsize=11.5, color="#333333", zorder=4)
arrow((87, 46), (95, 46), color=GREY, lw=1.6, ms=13, z=4)
ax.text(98, 46, "Other lower→higher channels\n(predicted “+”)", ha="left", va="center",
        fontsize=11.5, color="#333333", zorder=4)
ax.plot([87, 95], [35.5, 35.5], ls=(0, (4, 3)), color=GREY, lw=2.0, zorder=4)
ax.text(98, 35.5, "Same-level pairs (2):\nno directional prediction", ha="left",
        va="center", fontsize=11.5, color="#333333", zorder=4)

fig.savefig("/home/claude/paper4_work/figures/F0b_pipeline_hierarchy.png",
            bbox_inches="tight", facecolor="white")
print("saved")
