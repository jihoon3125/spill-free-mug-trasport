"""Plot multi-speed ablation: trajectory time budget vs spill ratio for each method."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "minjerk":     ("#d95f02", "Min-jerk (naive)", "o-"),
    "chomp":       ("#7570b3", "CHOMP (smooth)", "s-"),
    "stomp_spill": ("#e7298a", "STOMP + spill (gradient-free)", "v-"),
    "chomp_spill": ("#1b9e77", "CHOMP + spill (ours)", "D-"),
}


def main():
    d = np.load(ROOT / "results/multispeed.npz")
    Ts = d["Ts"]
    theta_max = float(d["theta_max"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel, title in [
        (axes[0], "spill", "Spill ratio (% of trajectory)",
         "Spill ratio vs trajectory time budget"),
        (axes[1], "maxtilt", "Max tilt of effective gravity (deg)",
         "Max tilt vs trajectory time budget"),
    ]:
        for m, (c, lbl, fmt) in PALETTE.items():
            vals = d[f"{m}_{key}"]
            if key == "spill":
                vals = vals * 100
            ax.plot(Ts, vals, fmt, color=c, label=lbl, lw=1.7, markersize=7)
        ax.set_xlabel("Trajectory total time T (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if key == "maxtilt":
            ax.axhline(theta_max, ls="--", color="r", lw=1, label=f"spill threshold {theta_max:.0f}°")
        if key == "spill":
            ax.set_ylim(-3, 105)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Multi-speed ablation (same 70 cm + 15 cm path, varying T)",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    out = ROOT / "figs/fig_multispeed.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
