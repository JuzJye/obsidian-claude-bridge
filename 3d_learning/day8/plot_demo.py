"""Plot the saved Day 8 experiment; no geometry recomputation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day8"
a = np.load(out / "arrays.npz")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].scatter(a["uv"][:, 0], a["uv"][:, 1], c=["C0", "C1", "C2"], s=80)
for i, uv in enumerate(a["uv"]):
    axes[0].annotate(f"P{i+1}", uv + [5, -5])
axes[0].set(xlim=(0,640), ylim=(480,0), xlabel="u (px)", ylabel="v (px)", title="Projection: 3D points to pixels")
for name, key in [("camera", "T_bc"), ("ee", "T_be"), ("TCP", "T_bt")]:
    T = a[key]
    axes[1].scatter(T[0,3], T[2,3], s=45)
    axes[1].annotate(name, (T[0,3],T[2,3]), xytext=(5,5), textcoords="offset points")
for i, pb in enumerate(a["pb"]):
    axes[1].plot([a["T_bc"][0,3],pb[0]], [a["T_bc"][2,3],pb[2]], "--", color=f"C{i}")
    axes[1].scatter(pb[0],pb[2], color=f"C{i}")
axes[1].set(xlabel="base x (m)", ylabel="base z (m)", title="Same observations in robot base (x-z slice)")
axes[1].set_aspect("equal", adjustable="datalim")
for ax in axes: ax.grid(alpha=.25)
fig.savefig(out/"coordinate-chain.png", dpi=170); plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
axes[0].bar(["P1","P2","P3"],np.linalg.norm(a["wrong"]-a["correct"],axis=1))
axes[0].set(ylabel="3D error (m)",title="Only change: mm interpreted as m")
x=np.arange(3)
axes[1].bar(x-.18,a["range_error"]*1000,.35,label="range used as z")
axes[1].bar(x+.18,a["stale_K_error"]*1000,.35,label="resize without updating K")
axes[1].set(xticks=x,xticklabels=["P1","P2","P3"],ylabel="3D error (mm)",title="Two distinct contract errors")
axes[1].legend(); fig.savefig(out/"contract-errors.png",dpi=170); plt.close(fig)
print("Saved coordinate-chain.png and contract-errors.png")
