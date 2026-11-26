"""D26: geometric constraints in a tiny rendered field and candidate scoring.
Independent CPU toy experiment, not NeuS, a full NeRF, MPC deployment, or Dreamer.
Run from the vault root: python -B 3d_learning/day26/demo.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/day26-mpl")
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day26"
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def render(center):
    # Unit ray; t and bin width are metres; density is 1/metre.
    t = (torch.arange(512)+.5)*(3./512)+1.
    sigma = 80.*torch.exp(-.5*((t-center)/.03)**2)
    optical = sigma*(3./512)
    trans = torch.exp(-torch.cat([torch.zeros(1), optical.cumsum(0)[:-1]]))
    weights = trans*(-torch.expm1(-optical))
    rgb = weights.sum()*torch.tensor([.2,.6,.9]) + torch.exp(-optical.sum())
    depth = (weights*t).sum()/weights.sum()
    return rgb, depth, sigma, t


def fit(use_depth, target_rgb, target_depth):
    center = torch.tensor(1.5, requires_grad=True)
    opt = torch.optim.Adam([center], lr=.03)
    history = []
    for step in range(180):
        rgb, depth, _, _ = render(center)
        color_loss = (rgb-target_rgb).square().mean()
        loss = color_loss + (100.*(depth-target_depth).square() if use_depth else 0.)
        opt.zero_grad(); loss.backward(); opt.step()
        history.append([step,float(center.detach()),float(color_loss.detach()),
                        float((depth-target_depth).abs().detach())])
    rgb,depth,sigma,t = render(center.detach())
    return dict(center_m=float(center.detach()), conditional_depth_m=float(depth),
                color_mse=float((rgb-target_rgb).square().mean()),
                depth_abs_error_m=float((depth-target_depth).abs())), np.array(history), sigma.numpy()


def segment_clearance(path, center, obstacle_radius, robot_radius=.04):
    # Exact minimum for each straight segment, not merely sampled waypoints.
    p, v = path[:-1], np.diff(path, axis=0)
    alpha = np.clip(np.sum((center-p)*v,axis=1)/np.sum(v*v,axis=1),0,1)
    closest = p+alpha[:,None]*v
    return float(np.min(np.linalg.norm(closest-center,axis=1))-obstacle_radius-robot_radius)


def score_candidates():
    amplitudes = np.arange(0.,.651,.05)
    x = np.linspace(0,1.2,25)
    paths = np.array([np.c_[x,a*np.sin(np.pi*x/1.2)] for a in amplitudes])
    center = np.array([.6,0.])
    dt = .1
    controls = np.diff(paths,axis=1)/dt  # m/s, integrator p_next=p+dt*u
    smooth_cost = np.sum(np.diff(controls,axis=1)**2,axis=(1,2))
    length = np.linalg.norm(np.diff(paths,axis=1),axis=2).sum(1)
    rows=[]
    for map_radius in [.18,.25]:
        clearance=np.array([segment_clearance(p,center,map_radius) for p in paths])
        # Costs are normalized by specified reference units, so sum is dimensionless.
        cost=length/1.2+.02*smooth_cost/(1.**2)
        feasible=(clearance>=.01)&(np.linalg.norm(controls,axis=2).max(1)<=1.)
        masked=np.where(feasible,cost,np.inf)
        if not np.any(feasible):
            raise RuntimeError("No feasible candidate; must stop or replan")
        best=int(np.argmin(masked))
        rows.append(dict(map_obstacle_radius_m=map_radius, feasible_candidates=int(feasible.sum()),
            chosen_amplitude_m=float(amplitudes[best]), chosen_index=best,
            predicted_clearance_m=float(clearance[best]),
            true_clearance_m=segment_clearance(paths[best],center,.25),
            first_velocity_m_s=controls[best,0].tolist(), cost=float(masked[best])))
    return rows,paths


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    np.random.seed(26);torch.manual_seed(26)
    target_rgb,target_depth,target_sigma,t=render(torch.tensor(2.6))
    target_rgb,target_depth=target_rgb.detach(),target_depth.detach()
    rows,arrays={},{}
    for name,depth_on in [("rgb_only",False),("rgb_plus_depth",True)]:
        rows[name],arrays[name+"_history"],arrays[name+"_sigma"]=fit(depth_on,target_rgb,target_depth)
    planner,paths=score_candidates()
    result=dict(seed=26,scope="one-ray parametric density fit plus one finite-horizon planning call",
        units="metres, seconds, m/s; renderer density 1/m; RGB [0,1]",
        versions=dict(numpy=np.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__),
        reference_density_center_m=2.6, reference_conditional_depth_m=float(target_depth),
        reconstruction=rows,planning=planner,
        boundary="No closed-loop MPC rollout, learned policy, or neural scene network was trained.")
    assert rows["rgb_plus_depth"]["depth_abs_error_m"]<.001
    assert rows["rgb_only"]["depth_abs_error_m"]>1.
    assert planner[0]["true_clearance_m"]<0<planner[1]["true_clearance_m"]
    (OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez(OUT/"arrays.npz",**arrays,t=t.numpy(),target_sigma=target_sigma.numpy(),paths=paths)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for name,c in [("rgb_only","#d87967"),("rgb_plus_depth","#248b89")]:
        axs[0].plot(t.numpy(),arrays[name+"_sigma"],label=name,c=c)
        axs[1].semilogy(arrays[name+"_history"][:,0],arrays[name+"_history"][:,3],label=name,c=c)
    axs[0].plot(t.numpy(),target_sigma.numpy(),"--",label="Reference",c="#315681")
    axs[0].set(xlabel="Ray distance / m",ylabel="Density / (1/m)",title="Same RGB does not identify the surface")
    axs[1].set(xlabel="Adam step",ylabel="Conditional-depth error / m",title="A trusted depth observation constrains location")
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/"geometry_constraint.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,row in zip(axs,planner):
        for p in paths:ax.plot(p[:,0],p[:,1],c="gray",alpha=.18)
        ax.add_patch(plt.Circle((.6,0),.29,color="#d87967",alpha=.2,label="True obstacle + robot radius"))
        ax.add_patch(plt.Circle((.6,0),row["map_obstacle_radius_m"]+.04,fill=False,
                              ls="--",edgecolor="#315681",label="Map obstacle + robot radius"))
        p=paths[row["chosen_index"]];ax.plot(p[:,0],p[:,1],c="#248b89",lw=2.5,label="Chosen candidate")
        ax.set(xlabel="World x / m",ylabel="World y / m",aspect="equal",
               title=f"Map radius {row['map_obstacle_radius_m']:.2f} m\nTrue clearance {1000*row['true_clearance_m']:.1f} mm")
        ax.legend(fontsize=7,loc="upper left");ax.set_ylim(-.35,.9)
    fig.suptitle("Identical candidates and score; only obstacle radius changes")
    fig.savefig(OUT/"collision_scoring.png",dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
