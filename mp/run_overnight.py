"""Autonomous overnight pipeline: scale grasps (A) + gamma/fill-level study (B),
regenerate figures, and rebuild the report. Robust: each stage is wrapped; a
failing stage is logged and the pipeline continues. Logs to docs/overnight_log.md.
"""
import subprocess, sys, time, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "docs/overnight_log.md"


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"- [{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def stage(name, cmd, timeout=7200):
    log(f"**START** {name}: `{' '.join(cmd)}`")
    t0 = time.time()
    try:
        r = subprocess.run([PY] + cmd, cwd=str(ROOT), capture_output=True,
                           text=True, timeout=timeout)
        dt = time.time() - t0
        tail = "\n".join(l for l in r.stdout.splitlines()
                         if any(k in l for k in ("saved", "aggregate", "monotonic", "T*",
                                                 "reachable", "spill", "feasible", "gamma=",
                                                 "theta=", "corr")))[-1500:]
        if r.returncode == 0:
            log(f"**OK** {name} ({dt/60:.1f} min)\n```\n{tail[-1200:]}\n```")
        else:
            log(f"**FAIL** {name} (rc={r.returncode})\n```\n{r.stderr[-1200:]}\n```")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"**TIMEOUT** {name} after {timeout/60:.0f} min")
        return False
    except Exception as e:
        log(f"**ERROR** {name}: {e}")
        return False


def main():
    with open(LOG, "a") as f:
        f.write(f"\n\n# Overnight run {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
    log("Plan: A (scale grasps) -> B (gamma pareto + fill level) -> figures -> report")

    # ---- A: scale up ----
    stage("A1 reachability (120 grasps)", ["mp/run_reachability.py", "--per-tax", "40"])
    stage("A2 method comparison (T=1.0, up to 80 grasps)",
          ["mp/run_method_comparison.py", "--T", "1.0", "--max-grasps", "80"])
    stage("A3 T* sweep", ["mp/run_tstar.py"])

    # ---- B: studies ----
    stage("B1 gamma pareto", ["mp/run_gamma_pareto.py"])
    stage("B2 fill level", ["mp/run_filllevel.py"])

    # ---- figures ----
    stage("figs extra (cone/accel/reach/feasibility)", ["mp/make_extra_figs.py"], timeout=1800)
    stage("fig trajectories", ["mp/make_fig_trajectories.py"], timeout=1800)

    # ---- report ----
    stage("build report docx", ["docs/_scripts/build_report.py"], timeout=600)
    log("converting report to PDF ...")
    try:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                        "--outdir", "docs", "docs/report.docx"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        log("**DONE** report.pdf rebuilt. Overnight pipeline complete.")
    except Exception as e:
        log(f"PDF convert error: {e}")


if __name__ == "__main__":
    main()
