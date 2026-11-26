from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day10"
a=np.load(out/"arrays.npz");r=json.loads((out/"results.json").read_text())
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
axes[0,0].scatter(*a["noisy_uv"].T,s=12,label="2 px noisy observations")
axes[0,0].scatter(*a["reproject"].T,s=20,marker="+",label="fitted reprojection")
axes[0,0].invert_yaxis();axes[0,0].set(title="PnP: 3D-2D residual",xlabel="u (px)",ylabel="v (px)");axes[0,0].legend(fontsize=8)
axes[0,1].imshow(np.ma.masked_where(~a["valid"],a["disparity"]),vmin=11,vmax=13,cmap="viridis")
axes[0,1].set(title="BM disparity: valid evaluation ROI (around 12 px)",xlabel="u (px)",ylabel="v (px)")
axes[1,0].plot(a["textured_cost"],"o-",label="textured patch")
axes[1,0].axvline(12,ls="--",color="gray");axes[1,0].set(title="SSD cost with texture",xlabel="Candidate disparity (px)",ylabel="Mean squared intensity error")
axes[1,1].plot(a["flat_cost"],"o-");axes[1,1].set(ylim=(-1,1),title="Only change: constant texture (24 tied minima)",xlabel="Candidate disparity (px)",ylabel="Mean squared intensity error")
fig.savefig(out/"pnp-stereo.png",dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
ang=np.linspace(0,2*np.pi,200);unit=np.stack([np.cos(ang),np.sin(ang)])
for key,label in [("J","regular pose"),("Jn","near singular")]:
    xy=a[key]@unit;axes[0].plot(*xy,label=label)
axes[0].set_aspect("equal");axes[0].set(title="Joint unit circle mapped by J",xlabel="dx (m / rad)",ylabel="dy (m / rad)");axes[0].legend(fontsize=8)
scales=[x["scale"] for x in r["local_linearization"]];errors=[x["linearization_error_m"] for x in r["local_linearization"]]
axes[1].loglog(scales,errors,"o-");axes[1].set(title="Local model error shrinks quadratically",xlabel="Joint increment scale",ylabel="Linearization error (m)");axes[1].grid(alpha=.3)
x=np.arange(2);axes[2].bar(x-.18,a["pinv"],.35,label="pseudoinverse");axes[2].bar(x+.18,a["dls"],.35,label="DLS lambda=0.05")
axes[2].set(xticks=x,xticklabels=["q1","q2"],ylabel="Proposed joint increment (rad)",title="Same 1 cm target, very different steps");axes[2].legend(fontsize=8)
fig.savefig(out/"jacobian-damping.png",dpi=160);plt.close(fig)
print("Saved pnp-stereo.png and jacobian-damping.png")
