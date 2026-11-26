"""Render saved D13 results. Does not rerun the experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day13"
    ap.add_argument("--input", type=Path, default=default_dir / "results.json")
    ap.add_argument("--output-dir", type=Path, default=default_dir)
    args = ap.parse_args()
    r = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc",
                         "axes.facecolor": "#ffffff", "savefig.facecolor": "#f8fafc"})
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    for b, color in [(.06, "#2563eb"), (.12, "#d97706")]:
        rows = [x for x in r["stereo_noise"] if x["baseline_m"] == b]
        z = [x["depth_m"] for x in rows]
        ax[0].plot(z, [1000*x["sigma_z_linear_m"] for x in rows],
                   "--", color=color, label=f"B={b*1000:.0f} mm: linear prediction")
        ax[0].plot(z, [1000*x["sigma_z_empirical_m"] for x in rows],
                   "o", ms=7, color=color, label=f"B={b*1000:.0f} mm: Monte Carlo")
    ax[0].set(xlabel="True optical depth Z (m)", ylabel="Depth standard deviation (mm)",
              title="Depth uncertainty grows roughly as Z squared")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=.2)
    u = r["uncertainty"]
    samples = 1000*np.asarray(u["xz_error_samples_m"])
    ax[1].scatter(samples[:, 0], samples[:, 1], s=9, alpha=.25, color="#2563eb",
                  label="First 600 synthetic errors")
    cov = 1e6*np.asarray(u["cov_xz_linear_m2"])
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    theta = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ell = Ellipse((0, 0), 4*np.sqrt(values[0]), 4*np.sqrt(values[1]),
                  angle=theta, fill=False, lw=2, color="#d97706",
                  label="Linear covariance, Mahalanobis radius 2")
    ax[1].add_patch(ell)
    ax[1].set(xlabel="X error (mm)", ylabel="Z error (mm)",
              title="Uncertainty is correlated, not an isotropic ball")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=.2)
    fig.suptitle("D13 | Ideal rectified stereo, f=600 px; synthetic Gaussian pixel noise",
                 fontsize=14)
    fig.savefig(args.output_dir / "depth-uncertainty.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    rows = r["frequency_cases"]
    hz = [x["hz"] for x in rows]
    ax[0].plot(hz, [1000*x["velocity_endpoint_m"] for x in rows],
               "o-", color="#2563eb", label="0.02 m/s, integrate actual dt")
    ax[0].plot(hz, [1000*x["fixed_delta_endpoint_m"] for x in rows],
               "s-", color="#dc2626", label="Repeat 0.002 m at every tick")
    ax[0].set(xlabel="Command frequency (Hz)", ylabel="Displacement after 1 second (mm)",
              title="Fixed per-step delta changes meaning with frequency")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=.2)
    rows = r["timestamp_cases"]
    ax[1].plot([1000*x["age_s"] for x in rows],
               [1000*x["position_error_m"] for x in rows], "o-", color="#7c3aed")
    ax[1].set(xlabel="Pose timestamp mismatch (ms)", ylabel="Base-frame point error (mm)",
              title="Wrong transform time produces a moving false target")
    ax[1].text(.04, .93, "Stationary target; camera translates at 0.1 m/s.\n"
               "Use camera pose at image capture time.",
               transform=ax[1].transAxes, va="top", fontsize=10)
    ax[1].grid(alpha=.2)
    fig.suptitle("D13 | Ideal translation examples; no robot or actuator dynamics", fontsize=14)
    fig.savefig(args.output_dir / "action-timing.png", dpi=170)
    plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()
