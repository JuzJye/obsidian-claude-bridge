"""Plot measured D23 optimizer results and the partial-observation example."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/day23-mpl")
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day23"
d=np.load(OUT/"arrays.npz")
r=json.loads((OUT/"results.json").read_text())
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(2,2,figsize=(11,8),layout="constrained")
colors={"least_squares":"#d97969","huber":"#269786"}
for name,color in colors.items():
    h=d[name+"_history"]
    axs[0,0].semilogy(h[:,0],h[:,2],label=name,color=color)
axs[0,0].set(title="Clean withheld projections (16 observations)",xlabel="Attempted iteration",ylabel="Coordinate RMSE / px")
axs[0,0].legend();axs[0,0].grid(alpha=.2)
labels=["All training","Clean training","Clean heldout"]
keys=["train_all_coordinate_rmse_px","train_clean_coordinate_rmse_px","heldout_clean_coordinate_rmse_px"]
x=np.arange(3)
for i,(name,color) in enumerate(colors.items()):
    vals=[r["optimizers"][name][k] for k in keys]
    bars=axs[0,1].bar(x+(i-.5)*.34,vals,.34,label=name,color=color)
    axs[0,1].bar_label(bars,fmt="%.2f",fontsize=8)
axs[0,1].set(xticks=x,xticklabels=labels,ylabel="Coordinate RMSE / px",title="A higher all-training error can be healthier",ylim=(0,4.7))
for name,color in colors.items():
    err=np.linalg.norm(d[name+"_points"]-d["truth"],axis=1)
    axs[1,0].plot(np.arange(32),err*1000,".-",label=name,c=color)
axs[1,0].set(title="True 3D error is a separate metric",xlabel="Point ID",ylabel="Point error / mm")
axs[1,0].legend();axs[1,0].grid(alpha=.2)
ax=axs[1,1];ax.axis("off")
ax.text(0,.98,"What this run actually verifies",weight="bold",fontsize=13,va="top")
text=(f"4 cameras; 2 fixed metric anchors; 32 free points\n"
f"112 training observations; 10 corrupted; 16 held out\n\n"
f"Analytic Jacobian max error: {r['analytic_jacobian_max_abs_error']:.2e}\n"
f"Schur vs dense update: {r['schur_dense_step_max_abs_difference']:.2e}\n"
f"Similarity gauge pixel difference: {r['similarity_gauge_max_pixel_difference']:.2e}\n\n"
"Small synthetic graph, known K, good initialization.\n"
"45 attempted updates; no global-optimum claim.\n"
"No real-image feature extraction or COLMAP run.")
ax.text(0,.87,text,va="top",linespacing=1.7)
fig.suptitle("D23 | Handwritten anchored BA: robust loss versus ordinary LS",fontweight="bold")
fig.savefig(OUT/"ba_validation.png",dpi=180);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
time=np.array([-.1,0,.1])
for v,color in [(-.2,"#d97969"),(.2,"#269786")]:
    axs[0].plot(time,1000*v*time,"o-",label=f"Hidden velocity {v:+.1f} m/s",c=color)
axs[0].axvline(0,color="gray",ls=":")
axs[0].scatter([.1],[0],marker="x",s=100,c="#215d81",label="Position-only conditional mean")
axs[0].set(title="Same current observation, opposite futures",xlabel="Time relative to current / s",ylabel="Position / mm")
axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
dt=np.linspace(.01,.2,100)
axs[1].plot(dt,np.sqrt(2)*.001/dt,c="#215d81")
axs[1].scatter([.1],[np.sqrt(2)*.001/.1],c="#d97969")
axs[1].set(title="History helps; differencing also amplifies noise",xlabel="Observation interval / s",ylabel="Velocity noise standard deviation / (m/s)")
axs[1].text(.97,.93,"Independent position noise: σ = 1 mm\nσ_velocity = √2 σ / Δt",
 transform=axs[1].transAxes,ha="right",va="top")
axs[1].grid(alpha=.2)
fig.suptitle("D23 | Observation is not state",fontweight="bold")
fig.savefig(OUT/"partial_observation.png",dpi=180);plt.close(fig)
