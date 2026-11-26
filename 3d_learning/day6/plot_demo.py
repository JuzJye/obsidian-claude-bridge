#!/usr/bin/env python3
"""Render Day 6 recorded results. Requires matplotlib; no model downloads."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=root / "_assets/3d_learning/day6/results.json")
    parser.add_argument("--output", type=Path,
                        default=root / "_assets/3d_learning/day6")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    grid = data["bev"]
    pts = np.array(data["points_table_m"])
    fig = plt.figure(figsize=(17, 6))
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.16, top=0.81, wspace=0.40)
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(*pts.T, c=pts[:, 2], cmap="viridis", s=48, vmin=0, vmax=0.3)
    ax.set(xlabel="table x [m]", ylabel="table y [m]", zlabel="height [m]",
           xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(0, 0.4),
           title="A. 14 valid back-projected points")
    for column, key, title, vmax in [(2, "counts", "B. Return count (not occupancy)", 1),
                                     (3, "max_height_m", "C. Maximum measured height [m]", 0.3)]:
        ax = fig.add_subplot(1, 3, column)
        arr = np.array(grid[key], dtype=float)
        arr = np.ma.array(arr, mask=~np.array(grid["observed"]))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(arr, origin="lower", extent=grid["extent_m"],
                       cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set(xlabel="table x [m]", ylabel="table y [m]", title=title)
        fig.colorbar(im, ax=ax, shrink=0.72)
    fig.suptitle("RGB-D -> table frame -> BEV\nGray cells have no returns; free space is not established", fontsize=13)
    fig.savefig(args.output / "rgbd-bev.png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    rows = data["dynamics"]["rollout"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for key, label, style in [("true_x_m", "True trajectory", "-"),
                               ("fit_x_m", "Fitted model (same dynamics)", ":"),
                               ("biased_x_m", "Biased model, free rollout", "--")]:
        axes[0].plot(steps, [100 * r[key] for r in rows], style, marker="o", label=label)
    axes[0].set(title="A. Action sequence is identical", xlabel="prediction horizon [steps]", ylabel="position [cm]")
    for key, label in [("teacher_forced_error_m", "One step from the true state"),
                       ("rollout_error_m", "Free rollout from own prediction")]:
        axes[1].plot(steps, [1000 * r[key] for r in rows], marker="o", label=label)
    axes[1].set(title="B. Same biased model, different evaluation", xlabel="prediction horizon [steps]", ylabel="absolute position error [mm]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"Synthetic dynamics | fitted rho={data['dynamics']['rho_fit']:.2f}; injected rho bias={data['dynamics']['rho_bias']:+.2f}", fontsize=13)
    fig.savefig(args.output / "world-model-rollout.png")
    plt.close(fig)
    print("Wrote rgbd-bev.png and world-model-rollout.png")


if __name__ == "__main__":
    main()
