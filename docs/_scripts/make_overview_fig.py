#!/usr/bin/env python
"""Slide-2 big-picture diagram: my thesis (grasp generation) -> this project
(spill-free carry). Conference-clean schematic with real grasp thumbnails.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figs"))

NAVY = "#1a365d"; SLATE = "#2c3e50"; RED = "#c0392b"; DRED = "#a93226"
BLUE = "#2c6fbb"; GREEN = "#27865a"; GRAY = "#888888"
THESIS_BG = "#eef4fb"; PROJ_BG = "#fdf0ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def crop(img, thr=18):
    """Trim uniform background border (corner colour)."""
    rgb = img[..., :3] if img.shape[-1] == 4 else img
    bg = rgb[0, 0]
    mask = (np.abs(rgb - bg) * 255 > thr).any(-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1]


def chip(ax, cx, cy, w, h, text, ec=SLATE, fc="white", fs=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=SLATE)


def arrow(ax, p0, p1, col=GRAY, lw=1.8, ms=15, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, color=col, arrowstyle="-|>",
                 mutation_scale=ms, lw=lw, ls=ls))


def thumb(ax, path, cx, cy, h, label, lcol):
    img = crop(plt.imread(path))
    ar = img.shape[1] / img.shape[0]
    w = h * ar
    ax.imshow(img, extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2],
              aspect="auto", zorder=3)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                 ec="#cccccc", lw=1.0, zorder=4))
    # task tag, top-left
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy + h / 2 - 0.02), 0.0, 0.0))
    ax.text(cx - w / 2 + 0.02, cy + h / 2 + 0.16, label, fontsize=10.5,
            color="white", fontweight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc=lcol, ec="none"))
    return w


def draw_cup(ax, x, y, ang, col, s=0.16):
    a = np.radians(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([(-s, 0), (s, 0), (0.72 * s, 1.7 * s), (-0.72 * s, 1.7 * s)])
    pts = (R @ pts.T).T + np.array([x, y])
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=col, lw=1.5, zorder=5))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 6.0))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 6.0); ax.axis("off")

    # region backgrounds
    ax.add_patch(FancyBboxPatch((0.1, 0.25), 7.05, 5.25,
                 boxstyle="round,pad=0.02,rounding_size=0.1", fc=THESIS_BG, ec="none"))
    ax.add_patch(FancyBboxPatch((7.35, 0.25), 5.7, 5.25,
                 boxstyle="round,pad=0.02,rounding_size=0.1", fc=PROJ_BG, ec="none"))
    ax.text(3.62, 5.72, "MY THESIS  ·  task-conditional grasp generation",
            ha="center", fontsize=13, color=NAVY, fontweight="bold")
    ax.text(10.2, 5.72, "THIS PROJECT  ·  spill-free carry (motion planning)",
            ha="center", fontsize=13, color=DRED, fontweight="bold")

    # ---- thesis: inputs -> grasp generation -> 3 task->grasp thumbnails ----
    chip(ax, 1.15, 3.95, 1.7, 0.7, "object\nmesh", ec=BLUE)
    chip(ax, 1.15, 2.45, 1.95, 1.0, "task (language)\n\"pour\" /\n\"drink\" /\n\"hand over\"",
         ec=BLUE, fs=10)
    arrow(ax, (2.02, 3.85), (2.55, 3.35))
    arrow(ax, (2.13, 2.6), (2.55, 3.0))
    ax.add_patch(FancyBboxPatch((2.6, 2.55), 1.45, 1.4,
                 boxstyle="round,pad=0.04,rounding_size=0.12", fc="white", ec=BLUE, lw=2))
    ax.text(3.32, 3.25, "Grasp\ngeneration\n(reward\nmodel)", ha="center", va="center",
            fontsize=10.5, color=SLATE)

    thumbs = [("grasp_drinking.png", "drink", GREEN),
              ("grasp_pouring.png", "pour", BLUE),
              ("grasp_handover.png", "hand-over", "#8e44ad")]
    ys = [4.55, 3.05, 1.55]
    tx = 5.55
    for (fn, lbl, col), yy in zip(thumbs, ys):
        thumb(ax, os.path.join(FIGS, fn), tx, yy, 1.25, lbl, col)
        arrow(ax, (4.1, 3.25), (tx - 0.78, yy), col=col, lw=1.4, ms=12)
    ax.text(5.55, 0.55, "same mug, task -> different grasp", ha="center",
            fontsize=10, color=SLATE, style="italic")

    # ---- cross arrow: a grasp flows into this project ----
    arrow(ax, (6.35, 3.05), (7.55, 4.05), col=NAVY, lw=2.4, ms=18)
    ax.text(6.95, 3.75, "a grasp", fontsize=10.5, color=NAVY, fontweight="bold",
            rotation=38, ha="center")

    # ---- this project: inputs -> trajectory optimization -> trajectory ----
    chip(ax, 10.2, 4.55, 4.4, 0.7, "inputs:  grasp pose  +  start & goal EE poses",
         ec=DRED, fs=10.5)
    arrow(ax, (10.2, 4.18), (10.2, 3.95), col=GRAY)
    ax.add_patch(FancyBboxPatch((8.5, 2.95), 3.4, 1.0,
                 boxstyle="round,pad=0.04,rounding_size=0.12", fc="white", ec=RED, lw=2))
    ax.text(10.2, 3.45, "Spill-free trajectory optimization\n(CHOMP + spill cost — ours)",
            ha="center", va="center", fontsize=10.5, color=SLATE)
    arrow(ax, (10.2, 2.9), (10.2, 2.55), col=GRAY)

    # output trajectory icon
    sx, gx, yb = 8.7, 11.7, 1.6
    ax.plot(sx, yb, "o", color=GREEN, ms=11, zorder=6)
    ax.plot(gx, yb, "o", color=NAVY, ms=11, zorder=6)
    ax.text(sx, yb - 0.4, "start", ha="center", fontsize=9.5, color=GREEN)
    ax.text(gx, yb - 0.4, "goal", ha="center", fontsize=9.5, color=NAVY)
    t = np.linspace(0, 1, 50)
    px = sx + (gx - sx) * t
    py = yb + 0.9 * np.sin(np.pi * t)
    ax.plot(px, py, color=RED, lw=2.6, zorder=5)
    for tt in (0.18, 0.5, 0.82):
        xc = sx + (gx - sx) * tt
        yc = yb + 0.9 * np.sin(np.pi * tt)
        draw_cup(ax, xc, yc + 0.08, 3 * np.cos(np.pi * tt), RED)
    ax.text(10.2, 0.55, "no-spill SE(3) carry trajectory", ha="center",
            fontsize=10, color=DRED, fontweight="bold")

    fig.tight_layout()
    p = os.path.join(FIGS, "overview_thesis_project.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
