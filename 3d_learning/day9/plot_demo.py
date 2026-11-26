from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
out=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day9"
a=np.load(out/"arrays.npz")
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for i in range(5):
    y=a["u1"][i,1]
    axes[0].axhline(y,color=f"C{i}",alpha=.35)
    axes[0].plot([a["u1"][i,0],a["u2"][i,0]],[y,y],"o-",color=f"C{i}")
axes[0].set(xlabel="u (px)",ylabel="v (px)",ylim=(370,110),title="Rectified pair: corresponding points share a row")
for i,b in enumerate([.03,.12,.24]):
    axes[1].scatter(a["X"][:,2],np.abs(a["estimates"][i,:,2]-a["X"][:,2])*1000,s=10,alpha=.5,label=f"baseline {b} m")
axes[1].set(xlabel="True optical depth (m)",ylabel="Absolute depth error (mm)",title="Same pixel noise; only baseline changes")
axes[1].legend();fig.savefig(out/"stereo-triangulation.png",dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for r in [.1,.9]: axes[0].add_patch(Circle((0,0),r,fill=False,ls="--",color="gray"))
for i,q in enumerate(a["solutions"]):
    elbow=.5*np.array([np.cos(q[0]),np.sin(q[0])]); p=a["goal"]
    axes[0].plot([0,elbow[0],p[0]],[0,elbow[1],p[1]],"o-",lw=3,label=f"IK branch {i+1}")
axes[0].scatter(1,0,marker="x",s=70,color="red",label="unreachable (1,0)")
axes[0].set(xlim=(-1,1.15),ylim=(-1,1),xlabel="base x (m)",ylabel="base y (m)",title="Position-only 2R arm: two branches and annulus")
axes[0].set_aspect("equal");axes[0].legend(fontsize=8)
for i in range(2):axes[1].semilogy(a[f"h{i}"],"o-",label=f"initialization {i+1}")
axes[1].set(xlabel="Iteration",ylabel="Position residual (m)",title="Numerical IK: error-to-update loop")
axes[1].legend();axes[1].grid(alpha=.25);fig.savefig(out/"ik-branches.png",dpi=170);plt.close(fig)
print("Saved stereo-triangulation.png and ik-branches.png")
