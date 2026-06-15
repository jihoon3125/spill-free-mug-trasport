"""Generate figures for proposal slides + report.

Inputs: results/ablation.npz
Outputs to figs/:
  fig_tilt_timeseries.png   — tilt(t) for 3 methods overlaid + spill threshold
  fig_spill_ratio.png       — bar chart of spill ratio (3 methods)
  fig_mug_traj_3d.png       — mug center 3D position over trajectory (3 methods)
  fig_summary_table.png     — text summary table
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "minjerk":     ("#d95f02", "Min-jerk (naive)"),
    "chomp":       ("#7570b3", "CHOMP (smooth)"),
    "stomp_spill": ("#e7298a", "STOMP + spill (gradient-free)"),
    "chomp_spill": ("#1b9e77", "CHOMP + spill (ours)"),
}


def main():
    d = np.load(ROOT / "results/ablation.npz")
    dt = float(d['dt'])
    theta_max = float(d['theta_max_deg'])
    N = d['minjerk_tilt'].shape[0]
    t = np.arange(N) * dt

    # ----- fig 1: tilt time series -----
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for key in ["minjerk", "chomp", "stomp_spill", "chomp_spill"]:
        c, lbl = PALETTE[key]
        ax.plot(t, d[f"{key}_tilt"], color=c, label=lbl, lw=1.5)
    ax.axhline(theta_max, color="r", ls="--", lw=1, label=f"spill threshold θ_max={theta_max:.0f}°")
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Mug effective tilt (deg)")
    ax.set_title("Per-waypoint tilt of effective gravity vector vs mug axis")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_tilt_timeseries.png", dpi=160)
    plt.close(fig)
    print(f"saved {FIGS/'fig_tilt_timeseries.png'}")

    # ----- fig 2: spill-ratio bar -----
    methods = ["minjerk", "chomp", "stomp_spill", "chomp_spill"]
    ratios = []
    for key in methods:
        tilt = d[f"{key}_tilt"]
        ratios.append((tilt > theta_max).mean())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [PALETTE[k][0] for k in methods]
    labels = [PALETTE[k][1].replace(" + ", "\n+ ") for k in methods]
    bars = ax.bar(labels, [r * 100 for r in ratios], color=colors)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r*100:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Spill ratio (% of trajectory time)")
    ax.set_ylim(0, max(70, max(ratios) * 110))
    ax.set_title("Spill ratio (lower = better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_spill_ratio.png", dpi=160)
    plt.close(fig)
    print(f"saved {FIGS/'fig_spill_ratio.png'}")

    # ----- fig 3: 3D mug trajectory -----
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111, projection="3d")
    for key in methods:
        c, lbl = PALETTE[key]
        pm = d[f"{key}_pmug"]
        ax.plot(pm[:, 0], pm[:, 1], pm[:, 2], color=c, label=lbl, lw=1.5)
        ax.scatter(pm[0, 0], pm[0, 1], pm[0, 2], color=c, s=30, marker="o")
        ax.scatter(pm[-1, 0], pm[-1, 1], pm[-1, 2], color=c, s=40, marker="X")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Mug center position (start ●  goal ×)")
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=22, azim=-50)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_mug_traj_3d.png", dpi=160)
    plt.close(fig)
    print(f"saved {FIGS/'fig_mug_traj_3d.png'}")

    # ----- fig 4: summary table -----
    rows = []
    for key in methods:
        tilt = d[f"{key}_tilt"]
        rows.append([
            PALETTE[key][1],
            f"{(tilt > theta_max).mean()*100:.0f}%",
            f"{tilt.max():.1f}°",
            f"{tilt.mean():.1f}°",
        ])
    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    ax.axis("off")
    headers = ["Method", "Spill ratio", "Max tilt", "Mean tilt"]
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j, h in enumerate(headers):
        tbl[(0, j)].set_facecolor("#e0e0e0")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_summary_table.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIGS/'fig_summary_table.png'}")


if __name__ == "__main__":
    main()
