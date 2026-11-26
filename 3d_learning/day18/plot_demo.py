"""Plot D18 stored results. Requires NumPy and Matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def fk(q):
    q = np.asarray(q)
    elbow = .5*np.stack([np.cos(q[..., 0]), np.sin(q[..., 0])], axis=-1)
    tip = elbow+.4*np.stack([np.cos(q.sum(axis=-1)), np.sin(q.sum(axis=-1))], axis=-1)
    return elbow, tip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=(
        Path(__file__).resolve().parents[2]/"_assets/3d_learning/day18"))
    args = ap.parse_args()
    root = args.input_dir
    r = json.loads((root/"results.json").read_text(encoding="utf-8"))
    a = np.load(root/"arrays.npz")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc",
                         "savefig.facecolor": "#f8fafc"})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax[0, 0].imshow(a["rgb"])
    ax[0, 0].set(title="Synthetic image: cup / obstacles / missing patch",
                 xlabel="u (px)", ylabel="v (px)")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#b8bec8")
    for axis, key, title in [
        (ax[0, 1], "fine_height", "BEV max height, 2 cm cells"),
        (ax[1, 0], "biased_height", "Same pixels, depth scaled by 1.05"),
        (ax[1, 1], "coarse_height", "BEV max height, 10 cm cells"),
    ]:
        im = axis.imshow(a[key], extent=a["extent"], origin="lower",
                         vmin=-.06, vmax=.35, cmap=cmap, interpolation="nearest")
        axis.set(title=title, xlabel="Base x (m)", ylabel="Base y (m)")
        axis.set_aspect("equal")
    fig.colorbar(im, ax=ax.ravel().tolist(), label="Max observed z (m)",
                 shrink=.8, pad=.025)
    fig.suptitle("D18 | Gray BEV cells have no point sample; they are not proven free",
                 fontsize=13)
    fig.savefig(root/"representation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.7), constrained_layout=True)
    plan = r["planning"]
    for axis in ax:
        for x, y, radius in plan["obstacles_x_y_radius_m"]:
            axis.add_patch(Circle((x, y), radius, color="#f59e0b", alpha=.65))
            axis.add_patch(Circle((x, y), radius+plan["link_radius_m"],
                                 fill=False, ls="--", color="#b45309"))
        axis.plot(0, 0, "ks", ms=6)
        axis.set(xlim=(-.65, 1.), ylim=(-1., .8), xlabel="Base x (m)", ylabel="Base y (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=.15)
    for branch, color, name in zip(plan["ik_branches"], ["#2563eb", "#dc2626"],
                                   ["IK branch: clear", "IK branch: collision"]):
        elbow, tip = fk(branch["q_rad"])
        nodes = np.array([[0, 0], elbow, tip])
        ax[0].plot(nodes[:, 0], nodes[:, 1], "o-", lw=4, color=color, label=name)
    ax[0].set_title("Same endpoint, different whole-arm geometry")
    ax[0].legend(fontsize=9, loc="lower left")
    way = a["waypoints"]
    direct_q = np.linspace(way[0], way[-1], 201)
    _, tip = fk(direct_q)
    ax[1].plot(tip[:, 0], tip[:, 1], "--", color="#dc2626", label="Direct-edge tip trace")
    _, tip = fk(a["motion"][:, 1:3])
    ax[1].plot(tip[:, 0], tip[:, 1], color="#059669", lw=2, label="Via-waypoint tip trace")
    # An explicitly chosen intermediate direct-edge pose illustrates the sweep.
    for q, color, alpha, label in [
        (way[0], "#475569", .55, "Start arm"),
        (direct_q[100], "#dc2626", .9, "Intermediate direct arm"),
        (way[1], "#059669", .4, "Via arm"),
    ]:
        elbow, tip = fk(q)
        nodes = np.array([[0, 0], elbow, tip])
        ax[1].plot(nodes[:, 0], nodes[:, 1], "o-", lw=2, color=color,
                   alpha=alpha, label=label)
    ax[1].set_title("Clear endpoints do not imply a clear path")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Planar teaching model | dashed obstacle boundary includes 2 cm link radius",
                 fontsize=13)
    fig.savefig(root/"ik-path.png", dpi=160)
    plt.close(fig)

    motion = a["motion"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    timing = plan["timing"]
    for k, label in enumerate(["Joint position (rad)", "Velocity (rad/s)",
                               "Acceleration (rad/s²)"]):
        for j, color in enumerate(["#2563eb", "#d97706"]):
            ax[k].plot(motion[:, 0], motion[:, 1+2*k+j], color=color, label=f"Joint {j+1}")
            if k:
                limits = timing["velocity_limits_rad_s" if k == 1 else "accel_limits_rad_s2"]
                for sign in [-1, 1]:
                    ax[k].axhline(sign*limits[j], color=color, ls=":", alpha=.45)
        ax[k].axvline(timing["segments"][0]["duration_s"], color="#64748b", ls="--", alpha=.6)
        ax[k].set_ylabel(label)
        ax[k].grid(alpha=.15)
    ax[0].legend(ncol=2)
    ax[2].set_xlabel("Physical motion time (s)")
    fig.suptitle("Cubic timing: stop at the waypoint; acceleration can jump", fontsize=14)
    fig.savefig(root/"trajectory.png", dpi=160)
    plt.close(fig)
    print("Wrote representation.png, ik-path.png, trajectory.png")


if __name__ == "__main__":
    main()
