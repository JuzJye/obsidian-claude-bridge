"""D28: independent reconstruction/dynamics audit, synthetic data only.
Run: python -B 3d_learning/day28/demo.py
Distances are metres; symmetric Chamfer is half the sum of unsquared means.
"""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/day28-mpl")
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day28"


def surface_metrics(pred,truth,critical_mask,tolerance=.01):
    distances=np.linalg.norm(pred[:,None,:]-truth[None,:,:],axis=-1)
    accuracy=distances.min(1);completeness=distances.min(0)
    precision=float(np.mean(accuracy<tolerance));recall=float(np.mean(completeness<tolerance))
    return dict(chamfer_mean_m=float(.5*(accuracy.mean()+completeness.mean())),
        precision_at_10mm=precision,recall_at_10mm=recall,
        fscore_at_10mm=float(2*precision*recall/max(precision+recall,1e-12)),
        critical_region_recall=float(np.mean(completeness[critical_mask]<tolerance)),
        accuracy_mean_m=float(accuracy.mean()),completeness_mean_m=float(completeness.mean()))


def true_step(s,a,dt=.1):
    acceleration=np.clip(a,-1,1)
    v=.95*s[...,1]+dt*acceleration
    return np.stack([s[...,0]+dt*v,v],-1)


def design(s,a):
    return np.c_[s,np.asarray(a),np.ones(len(s))]


def simulate(s,actions,model=None,teacher=False):
    true=s.copy();pred=s.copy();errors=[]
    for a in actions.T:
        nxt=true_step(true,a)
        if model is None:
            estimate=true_step(pred,a)
        else:
            estimate=design(true if teacher else pred,a)@model
        errors.append(np.sqrt(np.mean((estimate[:,0]-nxt[:,0])**2)))
        true,pred=nxt,estimate
    return np.array(errors)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(28)
    x,y=np.meshgrid(np.linspace(-.5,.5,30),np.linspace(-.5,.5,30))
    plane=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
    x,y=np.meshgrid(np.linspace(-.04,.04,7),np.linspace(-.04,.04,7))
    obstacle=np.c_[x.ravel(),y.ravel(),np.full(x.size,.85)]
    truth=np.r_[plane,obstacle]
    critical=np.r_[np.zeros(len(plane),bool),np.ones(len(obstacle),bool)]
    # A is precise where it reconstructs, but omits a collision-critical region.
    pred_a=plane+rng.normal(0,.0003,plane.shape)
    pred_b=truth+rng.normal(0,.001,truth.shape)
    reference=np.full((96,96,3),.6);reference[46:50,46:50]=.1
    image_a=np.full_like(reference,.6)
    image_b=np.clip(reference+rng.normal(0,.035,reference.shape),0,1)
    rows={}
    for name,pred,img in [("A_incomplete",pred_a,image_a),("B_complete_noisy",pred_b,image_b)]:
        rows[name]=surface_metrics(pred,truth,critical)
        rows[name]["image_psnr_db"]=float(-10*np.log10(np.mean((img-reference)**2)))
    # Exact projection-scale invariance; image fit cannot determine metric scale.
    X=np.array([[.1,.2,1.5],[-.2,.1,2.],[.3,-.1,2.5]])
    t=np.array([.1,0,.05])
    project=lambda q:600*q[:,:2]/q[:,2:3]+np.array([320,240])
    scale_error=float(np.max(np.abs(project(X+t)-project(2*(X+t)))))
    # Train on independent one-step samples in action range [-.5,.5].
    s=rng.uniform([-.3,-.2],[.3,.2],(1200,2));a=rng.uniform(-.5,.5,1200)
    y=true_step(s,a)+rng.normal(0,.0005,(len(s),2))
    weight=np.linalg.lstsq(design(s,a),y,rcond=None)[0]
    starts=rng.uniform([-.2,-.1],[.2,.1],(120,2))
    actions_id=rng.uniform(-.5,.5,(120,20))
    # Same initial states, same action signs/timing; only action amplitude changes.
    actions_ood=4.*actions_id
    curves={}
    for name,actions in [("in_distribution",actions_id),("outside_action_range",actions_ood)]:
        curves[name+"_rollout"]=simulate(starts,actions,weight)
        curves[name+"_teacher"]=simulate(starts,actions,weight,teacher=True)
    # Fixed-direction stress test makes bias accumulation directly visible.
    coherent=np.full((120,20),1.5)
    curves["constant_ood_rollout"]=simulate(starts,coherent,weight)
    # A 1 mm constant per-step position bias: one-step versus twenty-step checks.
    biased_weight=weight.copy();biased_weight[-1,0]+=.001
    curves["position_bias_rollout"]=simulate(starts,actions_id,biased_weight)
    dynamics={name:{str(h):float(curve[h-1]) for h in [1,5,10,20]} for name,curve in curves.items()}
    result=dict(seed=28,versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        metric_definition="Chamfer: half sum of unsquared directed means; metres",
        reference_points=len(truth),critical_points=len(obstacle),surface=rows,
        image_scope="Two diagnostic synthetic images, not renderings of the point clouds",
        scale2_projection_max_error_px=scale_error,
        dynamics=dict(dt_s=.1,training_action_range=[-.5,.5],true_acceleration_clip=[-1,1],
            train_samples=1200,test_trajectories=120,position_rmse_m=dynamics),
        protocol="Frozen model; no heldout feedback in rollout; teacher uses true previous state.")
    assert rows["A_incomplete"]["image_psnr_db"]>rows["B_complete_noisy"]["image_psnr_db"]
    assert rows["A_incomplete"]["critical_region_recall"]==0
    assert rows["B_complete_noisy"]["critical_region_recall"]==1
    assert scale_error<1e-12
    (OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez(OUT/"arrays.npz",truth=truth,pred_a=pred_a,pred_b=pred_b,critical=critical,
             reference_image=reference,image_a=image_a,image_b=image_b,**curves)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for ax,img,title in zip(axs[0],[reference,image_a,image_b],["Reference","A: high PSNR, missing small patch","B: lower PSNR, retains patch"]):
        ax.imshow(img);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        ax.add_patch(plt.Rectangle((44,44),8,8,fill=False,edgecolor="red",lw=1))
    for ax,pred,title in zip(axs[1,:2],[pred_a,pred_b],["A: missing raised obstacle","B: complete geometry"]):
        ax.scatter(truth[:,0],truth[:,2],c="#999999",s=7,label="Reference")
        ax.scatter(pred[:,0],pred[:,2],s=5,c="#248b89",label="Prediction")
        ax.set(title=title,xlabel="X / m",ylabel="Z / m",ylim=(.8,1.05));ax.legend(fontsize=7)
    ax=axs[1,2]
    names=["A","B"];vals=[rows[k]["critical_region_recall"] for k in rows]
    ax.bar(names,vals,color=["#d87967","#248b89"])
    ax.set(title="Critical-region recall at 10 mm",ylabel="Fraction",ylim=(0,1.15))
    fig.suptitle("Image and geometry are separate diagnostic fixtures, not a trained reconstruction system")
    fig.savefig(OUT/"reconstruction_audit.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    steps=np.arange(1,21)
    for key,label,c in [("in_distribution_rollout","ID rollout","#248b89"),
        ("outside_action_range_rollout","OOD rollout","#d87967"),
        ("outside_action_range_teacher","OOD teacher forced","#315681")]:
        axs[0].plot(steps,curves[key]*1000,label=label,c=c)
    for key,label,c in [("constant_ood_rollout","Constant OOD action","#d87967"),
                        ("position_bias_rollout","+1 mm per-step model bias","#315681")]:
        axs[1].plot(steps,curves[key]*1000,label=label,c=c)
    for ax in axs:ax.set(xlabel="Horizon / steps",ylabel="Position RMSE / mm");ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[0].set_title("Open-loop rollout versus teacher forcing")
    axs[1].set_title("Bias can accumulate coherently")
    fig.savefig(OUT/"dynamics_audit.png",dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
