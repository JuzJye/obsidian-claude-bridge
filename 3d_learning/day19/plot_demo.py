"""Render the saved Day19 experiments; numerical work remains in demo.py."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day19"
    a = np.load(out / "arrays.npz")
    r = json.loads((out / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    truth = a["camera_clean"]
    axes[0].scatter(*truth[:, :2].T, s=30, facecolors="none", edgecolors="0.6", label="Clean true points")
    scale_error = r["alignment"]["ordinary_ls"]["scale_relative_error_percent"]
    for key, label, color in [("ordinary_ls_points", f"LS: scale error {scale_error:.2f}%", "#ea580c"),
                              ("ransac_points", "RANSAC + inlier refit", "#2563eb")]:
        axes[0].scatter(*a[key][:, :2].T, s=9, color=color, label=label, alpha=.8)
    axes[0].set(xlabel="Camera x (m)", ylabel="Camera y (m)", title="Same 30% wrong correspondences; different fit")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    ids = np.arange(len(truth))
    for key, label, color in [("ordinary_ls_points", "LS residual", "#ea580c"),
                              ("ransac_points", "RANSAC residual", "#2563eb")]:
        residual = np.linalg.norm(a[key]-a["contaminated"], axis=1)*1000
        axes[1].scatter(ids, residual, s=12, alpha=.7, color=color, label=label)
    threshold_mm = r["data"]["ransac_threshold_m"]*1000
    axes[1].axhline(threshold_mm, color="black", ls="--", lw=1, label=f"{threshold_mm:g} mm consensus threshold")
    axes[1].scatter(ids[~a["known_inlier"]], np.full((~a["known_inlier"]).sum(), -6.),
                    marker="|", color="red", label="Known corrupted matches")
    axes[1].set(xlabel="Correspondence index", ylabel="Residual to supplied match (mm)",
                title="Rejected wrong matches need not fit well")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle("D19 | Robust Sim(3): physical error and contaminated-data loss are different")
    fig.savefig(out / "robust-alignment.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].scatter(*truth[:, :2].T, s=34, facecolors="none", edgecolors="0.6", label="Clean true points")
    for key, label, color in [("good_initialization", "Good init", "#2563eb"),
                              ("half_turn_initialization", "Half-turn init", "#ea580c")]:
        error_deg = r["icp"][key]["rotation_error_deg"]
        axes[0].scatter(*a[key+"_points"][:, :2].T, s=9, color=color,
                        label=f"{label}: {error_deg:.3f} deg pose error", alpha=.75)
        history = a[key+"_history"]*1000
        axes[1].semilogy(np.arange(len(history)), history, "o-", color=color,
                         label="Good initialization" if key.startswith("good") else "Half-turn initialization")
    axes[0].set(xlabel="Camera x (m)", ylabel="Camera y (m)", title="Same target, fixed metric scale, different initial pose")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Completed ICP updates (0 = initial pose)", ylabel="Nearest-neighbour RMSE (mm)",
                title="A decreasing objective can settle at a wrong pose")
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle("D19 | Point-to-point ICP: local improvement is not a global pose guarantee")
    fig.savefig(out / "icp-initialization.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 9), constrained_layout=True)
    circle = a["cylinder"][:72]
    axes[0,0].scatter(*circle[:,:2].T*1000, s=14, c="0.6", label="Unlabelled symmetric geometry")
    axes[0,0].plot([50,-50],[0,0],"o-",lw=1.5,color="#dc2626",label="Tracked semantic point: 100 mm apart")
    axes[0,0].annotate("Original ID",(50,0),xytext=(14,12),textcoords="offset points",ha="right")
    axes[0,0].annotate("Same ID at 180 deg",(-50,0),xytext=(2,-18),textcoords="offset points")
    axes[0,0].set(xlim=(-70,70),ylim=(-65,65),xlabel="Object x (mm)",ylabel="Object y (mm)",
                  title="Cylinder top view: geometry agrees, identity can differ")
    axes[0,0].set_aspect("equal")
    axes[0,0].legend(fontsize=8,loc="upper right")
    metrics = np.array([r["symmetry"]["ADD_m"],r["symmetry"]["ADD_S_m"]])*1000
    axes[0,1].bar(["ADD\nfixed point identities", "ADD-S\nGT to estimated nearest surface"], metrics,
                  color=["#ea580c","#2563eb"])
    axes[0,1].set(ylabel="Mean point distance (mm)",ylim=(0,120),title="180 deg pose difference; same sampled surface")
    axes[0,1].text(0,103,"100 mm",ha="center")
    axes[0,1].text(1,3,"~0 mm",ha="center")
    action=r["action"]
    origin=np.array(action["current_T_base_tcp"])[:2,3]*1000
    goal=np.array(action["absolute_base_target_m"])[:2]*1000
    wrong=np.array(action["wrong_left_translation_target_m"])[:2]*1000
    axes[1,0].scatter(*origin,color="black",label="Current TCP")
    for endpoint,label,color in [(goal,"Tool x +20 mm = base y +20 mm","#2563eb"),
                                  (wrong,"Wrong: unchanged vector used in base","#ea580c")]:
        axes[1,0].annotate("",xy=endpoint,xytext=origin,
                          arrowprops=dict(arrowstyle="->",lw=2,color=color))
        axes[1,0].scatter(*endpoint,color=color,label=label)
    axes[1,0].set(xlim=(390,435),ylim=(90,135),xlabel="Base x (mm)",ylabel="Base y (mm)",
                  title="Translation convention: compare equal-length commands")
    axes[1,0].set_aspect("equal")
    axes[1,0].legend(fontsize=7,loc="upper right")
    old=origin/1000
    rotated=np.array(action["pure_left_rotation_position_m"])[:2]
    theta=np.linspace(np.arctan2(old[1],old[0]),np.arctan2(old[1],old[0])+np.pi/2,80)
    axes[1,1].plot(np.linalg.norm(old)*np.cos(theta),np.linalg.norm(old)*np.sin(theta),"--",color="#ea580c")
    axes[1,1].scatter(0,0,c="black",marker="+",s=70,label="Base origin")
    axes[1,1].scatter(*old,c="#2563eb",s=50,label="Rotate at TCP: position unchanged")
    axes[1,1].scatter(*rotated,c="#ea580c",s=50,label="Pure left rotation: position orbits base")
    axes[1,1].set(xlim=(-.18,.48),ylim=(-.05,.49),xlabel="Base x (m)",ylabel="Base y (m)",
                  title="Same orientation change, different rotation pivot")
    axes[1,1].set_aspect("equal")
    axes[1,1].legend(fontsize=7,loc="center left")
    for ax in axes.ravel(): ax.grid(alpha=.15)
    fig.suptitle("D19 | Symmetry metrics and action conventions answer different questions")
    fig.savefig(out / "symmetry-actions.png",dpi=170)
    plt.close(fig)
    print("Saved robust-alignment.png, icp-initialization.png, symmetry-actions.png")


if __name__ == "__main__":
    main()
