"""Plot the recorded D22 data; run demo.py first."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/day22-mpl")
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day22"
d=np.load(OUT/"arrays.npz")
r=json.loads((OUT/"results.json").read_text())
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
ax=axs[0]
uv,pairs,keep=d["uv"],d["pairs"],d["keep"]
for k in np.linspace(0,len(pairs)-1,50,dtype=int):
    i,j=pairs[k]
    color="#269786" if keep[k] else "#d97969"
    ax.plot([uv[0,i,0],uv[1,j,0]+640],[uv[0,i,1],uv[1,j,1]],c=color,alpha=.6,lw=.7)
ax.scatter(uv[0,:,0],uv[0,:,1],s=5,c="#215d81")
ax.scatter(uv[1,:,0]+640,uv[1,:,1],s=5,c="#215d81")
ax.axvline(640,c="gray",ls=":");ax.invert_yaxis()
ax.set(title="Appearance is a proposal",xlabel="Two image canvases / px",ylabel="v / px")
ax.text(.02,.03,"180 tentative → 104 accepted\n104 true, 0 false (synthetic IDs)",transform=ax.transAxes,
        bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
truth=d["truth"][d["accepted"][:,0]];est=d["estimated"]
axs[1].scatter(truth[:,0],truth[:,2],s=12,c="#215d81",label="True tracks")
axs[1].scatter(est[:,0],est[:,2],s=10,c="#d97969",label="E + DLT")
axs[1].set(title="No BA yet: geometry remains biased",xlabel="World X / m",ylabel="World Z / m")
axs[1].legend(loc="lower right");axs[1].text(.02,.97,f"Median point error {r['triangulated_median_error_m']:.3f} m",
 transform=axs[1].transAxes,va="top")
base=np.array([x["baseline_m"] for x in r["baseline_control"]])
error=np.array([x["median_abs_depth_error_m"] for x in r["baseline_control"]])
axs[2].loglog(base,error,"o-",c="#d97969",label="Median |depth error| / m")
axs[2].set(title="Only baseline changes",xlabel="Baseline / m",ylabel="Median depth error / m")
axs[2].grid(alpha=.2,which="both")
axs[2].text(.03,.04,"Same scene and same pixel noise.\nReprojection stays ≈ 0.210 px.",transform=axs[2].transAxes)
fig.suptitle("D22 | Matching → geometry → conditioning",fontweight="bold")
fig.savefig(OUT/"sfm_pipeline.png",dpi=180);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
ax=axs[0]
c0=np.array([0.,0]);c1=np.array([.4,0]);p0=np.array([.2,4.]);p1=np.array([.3,4.])
fake=d["moving_reconstruction"][0,[0,2]]
for c,p in [(c0,fake),(c1,fake)]:
    ax.plot([c[0],p[0]],[c[1],p[1]],c="#999999",ls="--")
ax.scatter([0,.4],[0,0],marker="^",s=60,c="#215d81",label="Camera centers")
ax.scatter([p0[0],p1[0]],[4,4],c="#269786",s=45,label="Actual moving point")
ax.scatter(*fake,marker="x",s=90,c="#d97969",label="False static point")
ax.annotate("0.10 m motion",xy=p1,xytext=(.46,3.65),arrowprops=dict(arrowstyle="->"))
ax.annotate("Static fit: Z = 5.333 m\nTwo-view residual ≈ 0",xy=fake,xytext=(.46,5.1),
 arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="World X / m",ylabel="World Z / m",title="Motion can hide inside a static reconstruction",xlim=(-.05,.94),ylim=(-.25,6))
ax.legend(loc="lower right",fontsize=8)
ax=axs[1]
ax.bar(["a = -0.10 m","a = 0","a = +0.10 m"],[-.1,0,.1],color=["#d97969","#888888","#269786"])
ax.axhline(0,c="gray",lw=.8)
ax.set(title="A static map does not choose a future",ylabel="Next object displacement / m")
ax.text(.5,.97,"Toy deterministic transition: x_next = x + a\nSame current map; different commanded displacements.",
 transform=ax.transAxes,ha="center",va="top",fontsize=9)
ax.set_ylim(-.15,.16)
fig.suptitle("D22 | Static consistency and action-conditioned change",fontweight="bold")
fig.savefig(OUT/"static_and_dynamic.png",dpi=180);plt.close(fig)
