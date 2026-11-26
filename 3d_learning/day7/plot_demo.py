#!/usr/bin/env python3
"""Static figures for Day 7. Requires NumPy and Matplotlib."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "_assets/3d_learning/day7/results.json")
    parser.add_argument("--output", type=Path, default=root / "_assets/3d_learning/day7")
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x, y = np.array([0, 1, 2]), np.array([1, 2, 2])
    beta = report["qr"]["beta"]
    fitted = beta[0] + beta[1] * x
    axes[0].scatter(x, y, label="Observed y", zorder=3)
    axes[0].plot(x, fitted, label="QR least-squares fit")
    axes[0].vlines(x, fitted, y, color="#dc6450", linestyle="--", label="Residual")
    axes[0].set(title="A. Fit the data; retain the residual", xlabel="x (dimensionless)",
                ylabel="y (dimensionless)", xticks=[0, 1, 2])
    axes[0].legend()
    theta = np.linspace(0, 2*np.pi, 200)
    rotation = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    ellipse = rotation @ np.array([40*np.cos(theta), 10*np.sin(theta)])
    axes[1].plot(*ellipse, label="2-standard-deviation contour")
    for direction, extent, color, label in [(rotation[:, 0], 40, "#d86a33", "sigma = 20 mm"),
                                          (rotation[:, 1], 10, "#498bbd", "sigma = 5 mm")]:
        axes[1].plot([-extent*direction[0], extent*direction[0]],
                     [-extent*direction[1], extent*direction[1]], color=color, label=label)
    axes[1].set(title="B. Covariance eigenvectors and eigenvalues",
                xlabel="error x [mm]", ylabel="error y [mm]", aspect="equal")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(args.output / "qr-covariance.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    configs = [("correct_frame", "A. Correct frame, no obstacle"),
               ("yaw_error_90deg", "B. Extra 90-degree yaw in conversion"),
               ("straight_line_obstacle", "C. Correct frame, blocked straight path")]
    for ax, (name, title) in zip(axes.flat, configs):
        case = report["simulation"][name]
        for i, episode in enumerate(case["episodes"]):
            xy = np.array([r["position_m"] for r in episode["records"]])
            ax.plot(*xy.T, alpha=0.7, linewidth=1.3)
            ax.scatter(*xy[0], color="black", s=15, label="start" if i == 0 else None)
        ax.scatter(0.4, 0.3, color="red", marker="*", s=130, label="goal")
        if name == "straight_line_obstacle":
            ax.add_patch(Circle((0.2, 0.15), 0.05, color="gray", alpha=0.4))
            ax.text(0.2, 0.15, "obstacle", ha="center", va="center", fontsize=9)
        ax.set(title=f"{title}\nSuccess: {case['successes']}/{case['attempts']}",
               xlabel="base x [m]", ylabel="base y [m]", aspect="equal")
        ax.legend(fontsize=9)
    ax = axes[1, 1]
    for name, label in [("correct_frame", "Correct frame"),
                        ("yaw_error_90deg", "Yaw error"),
                        ("straight_line_obstacle", "Obstacle")]:
        records = report["simulation"][name]["episodes"][0]["records"]
        ax.plot([r["step"]*0.05 for r in records],
                [r["distance_m"] for r in records], label=label)
    ax.axhline(0.008, color="gray", linestyle=":", label="success threshold")
    ax.set(title="D. Same initial state: distance to goal",
           xlabel="simulated time [s]", ylabel="distance [m]")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Planar point-robot simulation | velocity control only; no trained policy or contact physics")
    fig.savefig(args.output / "simulation-cases.png")
    plt.close(fig)
    print("Wrote qr-covariance.png and simulation-cases.png")


if __name__ == "__main__":
    main()
