#!/usr/bin/env python
"""Schematic figures for the motion-planning term-project proposal.
Conference-paper style: clean, minimal, no chartjunk.
Outputs to docs/termproject_motionplanning/figs/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "figs"))
os.makedirs(OUT, exist_ok=True)

NAVY = "#1a365d"
SLATE = "#2c3e50"
RED = "#c0392b"
BLUE = "#2c6fbb"
GREEN = "#27865a"
GRAY = "#888888"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.0,
})


# ---------------------------------------------------------------------------
# Fig 1: spill condition — effective-gravity cone
# ---------------------------------------------------------------------------
def fig_spill_cone():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, (a_mag, title, ok) in zip(
        axes,
        [(0.0, "Static / slow  (a = 0)", True),
         (6.0, "Accelerating  (a != 0)", False)],
    ):
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-1.2, 4.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=13, color=SLATE, pad=10)

        # mug (a simple trapezoid cup), upright
        cup = Polygon([(-1.0, 0), (1.0, 0), (0.75, 2.2), (-0.75, 2.2)],
                      closed=True, fc="#eef2f7", ec=SLATE, lw=2)
        ax.add_patch(cup)
        # apparent "up" = a - g  (no-spill: this must stay within rim cone)
        g = np.array([0.0, -9.81])
        a = np.array([a_mag, 0.0])
        up = a - g
        up_dir = up / np.linalg.norm(up)

        # cup axis (vertical, +z of cup frame)
        ax.add_patch(FancyArrowPatch((0, 2.2), (0, 4.1), color=SLATE,
                     arrowstyle="-|>", mutation_scale=15, lw=2, ls=(0, (4, 3))))
        ax.text(-1.55, 3.7, "cup axis", color=SLATE, fontsize=11)

        # rim cone (allowed tilt before spill), half-angle theta_max
        th = np.radians(28)
        apex = np.array([0, 1.0])
        for s in (-1, 1):
            d = np.array([np.sin(s * th), np.cos(s * th)]) * 3.0
            ax.add_patch(FancyArrowPatch(apex, apex + d, color=GREEN,
                         arrowstyle="-", lw=1.6, ls="--"))
        ax.text(-1.2, 4.35, r"$\theta_{max}$ rim cone", color=GREEN, fontsize=11)

        # apparent-up vector drawn from apex
        col = GREEN if ok else RED
        ax.add_patch(FancyArrowPatch(apex, apex + up_dir * 2.9, color=col,
                     arrowstyle="-|>", mutation_scale=18, lw=2.8))
        tip = apex + up_dir * 2.9
        ax.text(tip[0] + 0.12, tip[1], r"$a-g$", color=col, fontsize=13)

        # acceleration arrow
        if a_mag > 0:
            ax.add_patch(FancyArrowPatch((1.2, 0.7), (2.5, 0.7), color=BLUE,
                         arrowstyle="-|>", mutation_scale=16, lw=2))
            ax.text(1.7, 0.85, r"$a$", color=BLUE, fontsize=13)

        verdict = "no spill  (in cone)" if ok else "SPILL  (exits cone)"
        ax.text(0, -0.95, verdict, ha="center", fontsize=12.5,
                color=col, fontweight="bold")

    fig.suptitle("No-spill condition:  angle( cup axis ,  a - g )  <  "
                 r"$\theta_{max}$", fontsize=14, color=NAVY, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "spill_cone.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# Fig 2: method pipeline (affordance grasp -> spill-aware traj opt)
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.6); ax.axis("off")

    boxes = [
        (0.3, "mug mesh\n+ task\n(\"carry water\")", "#eef2f7", SLATE),
        (3.1, "Reward model\n(thesis, Stage 1)\n-> grasp pose", "#e8f0fb", BLUE),
        (5.9, "grasp-conditioned\nspill cone\n" + r"$\theta_{max}(\,grasp\,)$", "#e9f6ee", GREEN),
        (8.7, "CHOMP +\nspill cost\n$J_{sm}+J_{obs}+J_{spill}$", "#fbeceb", RED),
        (11.0, "SE(3) EE\ntrajectory\n(no-spill)", "#eef2f7", SLATE),
    ]
    w = 2.3; h = 1.9; y = 0.9
    centers = []
    for x, txt, fc, ec in boxes:
        bx = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                            fc=fc, ec=ec, lw=2)
        ax.add_patch(bx)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=11.5, color=SLATE)
        centers.append((x, x + w))
    for i in range(len(centers) - 1):
        x0 = centers[i][1]; x1 = centers[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2),
                     color=GRAY, arrowstyle="-|>", mutation_scale=18, lw=2))
    ax.text(6.75, 0.25, "Our contribution: grasp choice sets the spill margin "
            r"$\theta_{max}$ that the optimizer plans against",
            ha="center", fontsize=12, color=NAVY, style="italic")
    fig.tight_layout()
    p = os.path.join(OUT, "method_pipeline.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# Fig 3: 3-way ablation schematic
# ---------------------------------------------------------------------------
def fig_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    titles = ["Baseline 1: Linear interp",
              "Baseline 2: CHOMP\n(smooth + obstacle)",
              "Ours: CHOMP + spill cost"]
    cols = [GRAY, BLUE, RED]
    spill = ["high spill", "medium spill", "no spill"]
    for ax, title, col, sp in zip(axes, titles, cols, spill):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        ax.set_title(title, fontsize=12, color=SLATE, pad=8)
        # obstacle
        ob = plt.Circle((5, 2.0), 1.1, fc="#dddddd", ec=GRAY, lw=1.5)
        ax.add_patch(ob)
        # start/goal
        ax.plot(1, 1.2, "o", color=GREEN, ms=10); ax.text(0.4, 0.3, "start", fontsize=10)
        ax.plot(9, 1.2, "o", color=NAVY, ms=10); ax.text(8.4, 0.3, "goal", fontsize=10)
        t = np.linspace(0, 1, 60)
        if col == GRAY:        # linear -> through obstacle, jerky tilt
            x = 1 + 8 * t; ypath = 1.2 + 0 * t
        elif col == BLUE:      # chomp -> avoids obstacle, smooth
            x = 1 + 8 * t; ypath = 1.2 + 2.6 * np.sin(np.pi * t)
        else:                  # ours -> avoids + keeps cup level
            x = 1 + 8 * t; ypath = 1.2 + 2.6 * np.sin(np.pi * t)
        ax.plot(x, ypath, color=col, lw=3)
        # cup glyphs along path
        for tt in (0.2, 0.5, 0.8):
            xc = 1 + 8 * tt; yc = 1.2 + (0 if col == GRAY else 2.6 * np.sin(np.pi * tt))
            if col == GRAY:
                ang = 35 * np.sin(8 * tt)   # wild tilt
            elif col == BLUE:
                ang = 18 * np.cos(np.pi * tt)  # some tilt
            else:
                ang = 4 * np.cos(np.pi * tt)   # near-level
            _draw_cup(ax, xc, yc + 0.55, ang, col)
        ax.text(5, 5.4, sp, ha="center", fontsize=12, color=col, fontweight="bold")
    fig.suptitle("3-way ablation — spill ratio / path length / smoothness / plan time",
                 fontsize=13, color=NAVY, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "ablation_plan.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def _draw_cup(ax, x, y, ang_deg, col):
    a = np.radians(ang_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([(-0.22, 0), (0.22, 0), (0.16, 0.5), (-0.16, 0.5)])
    pts = (R @ pts.T).T + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=col, lw=1.6))


# ---------------------------------------------------------------------------
# Fig 4: trajectory optimizer AS a grasp evaluator (Framing B)
# ---------------------------------------------------------------------------
def fig_grasp_evaluator():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

    grasps = [  # (y, label, half_angle_deg, grip, color, margin)
        (4.15, "handle grip", 32, "handle", GREEN, 32),
        (2.65, "rim grip",    22, "rim",    BLUE,  22),
        (1.15, "body grip",   12, "body",   RED,   12),
    ]

    def mini_cup(cx, cy, th_deg, grip, col):
        s = 0.42
        cup = Polygon([(cx - s, cy), (cx + s, cy),
                       (cx + 0.72 * s, cy + 1.0), (cx - 0.72 * s, cy + 1.0)],
                      closed=True, fc="#eef2f7", ec=SLATE, lw=1.6)
        ax.add_patch(cup)
        # rim cone (half-angle th) opening up from cup centre
        th = np.radians(th_deg); apex = np.array([cx, cy + 0.45])
        for sgn in (-1, 1):
            d = np.array([np.sin(sgn * th), np.cos(sgn * th)]) * 0.68
            ax.add_patch(FancyArrowPatch(apex, apex + d, color=col,
                         arrowstyle="-", lw=1.5, ls="--"))
        # grip marker
        gp = {"handle": (cx + s + 0.12, cy + 0.5),
              "rim": (cx + 0.62 * s, cy + 1.0),
              "body": (cx - s, cy + 0.28)}[grip]
        ax.plot(*gp, marker="o", ms=9, color=col, mec="white", mew=1.2, zorder=5)
        if grip == "handle":   # little handle arc
            from matplotlib.patches import Arc
            ax.add_patch(Arc((cx + s, cy + 0.5), 0.3, 0.6, angle=0,
                             theta1=-80, theta2=80, color=SLATE, lw=1.6))

    # left column: three grasps
    ax.text(1.35, 5.0, "candidate grasps\n(taxonomy / region)", ha="center",
            fontsize=11, color=SLATE)
    for y, lbl, th, grip, col, m in grasps:
        mini_cup(1.35, y - 0.5, th, grip, col)
        ax.text(2.55, y, lbl, fontsize=11, color=col, va="center", fontweight="bold")

    # planner box
    bx = FancyBboxPatch((4.35, 1.55), 2.5, 2.3,
                        boxstyle="round,pad=0.05,rounding_size=0.14",
                        fc="#fbeceb", ec=RED, lw=2)
    ax.add_patch(bx)
    ax.text(5.6, 2.7, "Spill-aware\nCHOMP\n(per grasp)", ha="center", va="center",
            fontsize=12, color=SLATE)
    for _, _, _, _, col, _ in grasps:
        pass
    for y, *_ in grasps:
        ax.add_patch(FancyArrowPatch((3.55, y - 0.0), (4.35, 2.7), color=GRAY,
                     arrowstyle="-|>", mutation_scale=13, lw=1.4))

    # bar chart: achievable spill margin per grasp
    bx0 = 7.6
    ax.add_patch(FancyArrowPatch((6.85, 2.7), (bx0 - 0.15, 2.7), color=GRAY,
                 arrowstyle="-|>", mutation_scale=15, lw=1.8))
    ax.text((bx0 + 12.6) / 2, 4.9, r"achievable spill margin  $\theta_{max}$  (score)",
            ha="center", fontsize=11.5, color=NAVY)
    ax.text((bx0 + 12.6) / 2, 4.45, "wider margin = easier, safer carry  (illustrative)",
            ha="center", fontsize=9.5, color=GRAY)
    maxw = 4.4
    for (y, lbl, th, grip, col, m), yy in zip(grasps, (3.9, 2.7, 1.5)):
        w = maxw * m / 34.0
        ax.add_patch(plt.Rectangle((bx0, yy - 0.28), w, 0.56, fc=col, ec="none",
                                   alpha=0.85))
        ax.text(bx0 + w + 0.15, yy, f"{m}", va="center", fontsize=11, color=col,
                fontweight="bold")
        ax.text(bx0 - 0.15, yy, lbl.split()[0], va="center", ha="right",
                fontsize=10, color=SLATE)
    # feedback arrow (future work) — contained in the bottom strip
    ax.add_patch(FancyArrowPatch((11.2, 0.5), (1.7, 0.5),
                 connectionstyle="arc3,rad=0.12", color=NAVY,
                 arrowstyle="-|>", mutation_scale=15, lw=1.6, ls=(0, (5, 4))))
    ax.text(6.4, 0.08, "future (thesis): spill margin -> grasp-generation reward "
            "(close the loop)", ha="center", fontsize=10.5, color=NAVY, style="italic")

    fig.tight_layout()
    p = os.path.join(OUT, "grasp_evaluator.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_spill_cone()
    fig_pipeline()
    fig_ablation()
    fig_grasp_evaluator()
    print("done ->", OUT)
