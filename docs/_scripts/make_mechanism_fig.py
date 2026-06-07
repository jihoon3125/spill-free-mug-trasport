#!/usr/bin/env python
"""Slide-5 figure: why the grasp changes the spill margin (offset / lever).
Same arm motion, different grasp offset r -> different cup acceleration."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Arc, Rectangle

FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figs"))
NAVY="#1a365d"; SLATE="#2c3e50"; RED="#c0392b"; BLUE="#2c6fbb"; GREEN="#27865a"; GRAY="#888888"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})

def cup(ax, x, y, col=SLATE, s=0.38):
    ax.add_patch(Polygon([(x-s,y),(x+s,y),(x+0.72*s,y+1.0),(x-0.72*s,y+1.0)],
                 closed=True, fc="#eef2f7", ec=col, lw=1.8))

def gripper(ax, x, y):
    ax.add_patch(Rectangle((x-0.28,y-0.1),0.56,0.16, fc=SLATE, ec="none"))
    ax.add_patch(Rectangle((x-0.28,y+0.06),0.12,0.42, fc=SLATE, ec="none"))
    ax.add_patch(Rectangle((x+0.16,y+0.06),0.12,0.42, fc=SLATE, ec="none"))
    ax.text(x, y-0.4, "EE / wrist", ha="center", fontsize=10, color=SLATE)

def panel(ax, cx, title, r_len, cup_dx, cup_dy, col):
    wx, wy = cx-1.5, 0.9
    gripper(ax, wx, wy)
    ccx, ccy = wx+cup_dx, wy+cup_dy
    cup(ax, ccx, ccy, col)
    # offset vector r
    ax.add_patch(FancyArrowPatch((wx, wy+0.5),(ccx, ccy),
                 color=col, arrowstyle="-|>", mutation_scale=15, lw=2, ls=(0,(4,2))))
    midx,midy=(wx+ccx)/2,(wy+0.5+ccy)/2
    ax.text(midx-0.15, midy+0.1, "r", color=col, fontsize=14, fontweight="bold")
    # same angular accel alpha at wrist
    ax.add_patch(Arc((wx,wy+0.55),0.9,0.9, angle=0, theta1=20, theta2=140, color=BLUE, lw=2.2))
    ax.add_patch(FancyArrowPatch((wx-0.32,wy+0.78),(wx-0.45,wy+0.62),color=BLUE,
                 arrowstyle="-|>", mutation_scale=12, lw=2))
    ax.text(wx-0.1, wy+1.15, r"$\alpha$ (same)", color=BLUE, fontsize=11, ha="center")
    # induced cup acceleration ~ alpha x r  (length grows with r)
    ax.add_patch(FancyArrowPatch((ccx, ccy+0.5),(ccx+ r_len, ccy+0.5),
                 color=RED, arrowstyle="-|>", mutation_scale=16, lw=2.6))
    ax.text(ccx+r_len+0.1, ccy+0.5, r"$a_{cup}$", color=RED, fontsize=12, va="center")
    ax.text(cx, 3.85, title, ha="center", fontsize=12, color=col, fontweight="bold")

def main():
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.set_xlim(0,11.5); ax.set_ylim(-0.2,4.2); ax.axis("off"); ax.set_aspect("equal")
    panel(ax, 3.0, "handle grip  ->  large offset r", 1.25, 0.9, 2.0, GREEN)
    ax.axvline(5.9, color="#dddddd", lw=1.2)
    panel(ax, 8.6, "body grip  ->  small offset r", 0.55, -0.2, 1.0, RED)
    ax.text(5.75, -0.05,
            r"$a_{cup} = a_{ee} + \alpha \times r + \omega \times (\omega \times r)$"
            "      same arm motion, different grasp offset r  ->  different acceleration at the liquid",
            ha="center", fontsize=12.5, color=NAVY)
    fig.tight_layout()
    p = os.path.join(FIGS, "grasp_mechanism.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", p)

if __name__ == "__main__":
    main()
