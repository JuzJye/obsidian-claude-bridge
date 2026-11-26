"""D27 independent CPU RGB-D backprojection, visibility-gated feature fusion,
and a linear action-conditioned prediction control. Not a V-JEPA implementation.
Run: python -B 3d_learning/day27/demo.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/day27-mpl")
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day27"
SIZE,F,CX=96,80.,47.5


def camera_image(camera_x):
    v,u=np.mgrid[:SIZE,:SIZE]
    # Parallel cameras; foreground plane z=1, background plane z=1.5.
    front_x=(u-CX)/F+camera_x
    front_y=(v-CX)/F
    foreground=(np.abs(front_x)<.18)&(np.abs(front_y)<.25)
    depth=np.where(foreground,1.,1.5)
    features=np.stack([~foreground,foreground],-1).astype(float)
    return depth,features


def scene_points():
    x,y=np.meshgrid(np.linspace(-.55,.55,41),np.linspace(-.28,.28,21))
    back=np.c_[x.ravel(),y.ravel(),np.full(x.size,1.5)]
    x,y=np.meshgrid(np.linspace(-.16,.16,17),np.linspace(-.23,.23,15))
    front=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
    return np.r_[back,front],np.r_[np.zeros(len(back),int),np.ones(len(front),int)]


def fuse(points,cameras,depths,features,gate=True,extrinsic_bias=0.):
    total=np.zeros((len(points),2)); counts=np.zeros(len(points))
    for cam,depth,feature in zip(cameras,depths,features):
        q=points-np.array([cam+extrinsic_bias,0,0])
        uv=np.rint(F*q[:,:2]/q[:,2:3]+CX).astype(int)
        inside=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<SIZE)&(uv[:,1]>=0)&(uv[:,1]<SIZE)
        u,v=np.clip(uv[:,0],0,SIZE-1),np.clip(uv[:,1],0,SIZE-1)
        valid=inside & ((np.abs(q[:,2]-depth[v,u])<.02) if gate else True)
        total+=valid[:,None]*feature[v,u]
        counts+=valid
    fused=total/np.maximum(counts[:,None],1)
    labels=np.where(counts>0,fused.argmax(1),-1)  # -1 means unobserved, not background.
    return labels,counts,fused


def true_step(state,action,dt=.2):
    x,v=state[...,0],state[...,1]
    return np.stack([x+dt*v+.5*dt*dt*action,.95*v+dt*action],-1)


def design(s,a,use_action):
    return np.c_[s,a,np.ones(len(s))] if use_action else np.c_[s,np.ones(len(s))]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(27)
    cameras=np.array([-.28,.28])
    depths,features=zip(*(camera_image(c) for c in cameras))
    depths,features=np.array(depths),np.array(features)
    points,truth=scene_points()
    naive,ncount,_=fuse(points,cameras,depths,features,False)
    gated,gcount,gfeat=fuse(points,cameras,depths,features,True)
    biased,bcount,_=fuse(points,cameras,depths,features,True,.05)
    common=(gcount>0)&(bcount>0)
    fusion=dict(points=len(points),visible_points=int((gcount>0).sum()),
        coverage=float(np.mean(gcount>0)),unknown_points=int((gcount==0).sum()),
        naive_accuracy_all=float(np.mean(naive==truth)),
        naive_accuracy_on_visible=float(np.mean(naive[gcount>0]==truth[gcount>0])),
        gated_accuracy_on_visible=float(np.mean(gated[gcount>0]==truth[gcount>0])),
        extrinsic_bias_m=.05,biased_coverage=float(np.mean(bcount>0)),
        common_visible_points=int(common.sum()),
        biased_accuracy_common=float(np.mean(biased[common]==truth[common])))
    # Real backprojection of synthetic depth images, noise level is the only variable.
    yy,xx=np.mgrid[0:SIZE:4,0:SIZE:4]
    pixel_noise=rng.normal(size=(2,)+xx.shape)
    backprojection=[]
    for sigma in [0.,.005,.02]:
        clouds=[];truths=[]
        for i,cam in enumerate(cameras):
            z=depths[i,yy,xx];zn=z+sigma*pixel_noise[i]
            rays=np.stack([(xx-CX)/F,(yy-CX)/F,np.ones_like(xx)],-1)
            clouds.append((rays*zn[...,None]+[cam,0,0]).reshape(-1,3))
            truths.append((rays*z[...,None]+[cam,0,0]).reshape(-1,3))
        cloud=np.concatenate(clouds); reference=np.concatenate(truths)
        voxel_keys=np.floor(cloud/.025).astype(int)
        voxels=len(np.unique(voxel_keys,axis=0))
        backprojection.append(dict(depth_noise_std_m=sigma,points=len(cloud),voxels_25mm=voxels,
            point_coordinate_rmse_m=float(np.sqrt(np.mean((cloud-reference)**2)))))
    # 80 independent trajectories; fit on first 56, evaluate last 24.
    states=[];actions=[];targets=[]
    for episode in range(80):
        s=rng.uniform([-.3,-.2],[.3,.2])
        ss=[];aa=[];tt=[]
        for _ in range(12):
            a=rng.uniform(-1,1)
            nxt=true_step(s,a)
            ss.append(s);aa.append(a);tt.append(nxt+rng.normal(0,.001,2))
            s=nxt
        states.append(ss);actions.append(aa);targets.append(tt)
    S,A,Y=np.array(states),np.array(actions),np.array(targets)
    train=(S[:56].reshape(-1,2),A[:56].ravel(),Y[:56].reshape(-1,2))
    test=(S[56:].reshape(-1,2),A[56:].ravel(),Y[56:].reshape(-1,2))
    controls=np.array([-1.,0.,1.]);query=np.zeros((3,2));predictions={}
    model_metrics={}
    for name,act in [("without_action",False),("with_action",True)]:
        weight=np.linalg.lstsq(design(train[0],train[1],act),train[2],rcond=None)[0]
        pred=design(test[0],test[1],act)@weight
        predictions[name]=design(query,controls,act)@weight
        model_metrics[name]=dict(test_position_rmse_m=float(np.sqrt(np.mean((pred[:,0]-test[2][:,0])**2))),
            test_velocity_rmse_m_s=float(np.sqrt(np.mean((pred[:,1]-test[2][:,1])**2))),
            probe_next_positions_m=predictions[name][:,0].tolist())
    result=dict(seed=27,versions=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        scope="analytic two-plane RGB-D and linear prediction; no learned video backbone",
        fusion=fusion,backprojection=backprojection,
        trajectory_split=dict(train=56,test=24,steps=12,dt_s=.2),prediction=model_metrics,
        probe_actions_m_s2=controls.tolist(),probe_true_next_positions_m=true_step(query,controls)[:,0].tolist())
    assert fusion["gated_accuracy_on_visible"]==1.
    assert model_metrics["with_action"]["test_position_rmse_m"]<.002
    (OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez(OUT/"arrays.npz",points=points,truth=truth,naive=naive,gated=gated,depths=depths,
             gated_count=gcount,features=features,actions=controls,**predictions)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    axs[0].imshow(features[0,...,1],cmap="coolwarm",vmin=0,vmax=1)
    axs[0].set(title="View 0: front plane feature",xlabel="u / pixel",ylabel="v / pixel")
    for ax,label,title in zip(axs[1:],[naive,gated],["No visibility test","Depth-gated; unknown is gray"]):
        back=points[:,2]>1.2
        colors=np.where(label[back]<0,"#999999",np.where(label[back]==0,"#315681","#d87967"))
        ax.scatter(points[back,0],points[back,1],c=colors,s=12)
        ax.set(title=title,xlabel="Back plane X / m",ylabel="Y / m",aspect="equal")
    fig.suptitle("Projecting inside an image does not prove visibility")
    fig.savefig(OUT/"feature_visibility.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for name,c in [("without_action","#d87967"),("with_action","#248b89")]:
        axs[0].plot(controls,1000*predictions[name][:,0],"o-",label=name,c=c)
    axs[0].plot(controls,1000*true_step(query,controls)[:,0],"k--",label="True")
    axs[0].set(xlabel="Acceleration command / (m/s^2)",ylabel="Next position / mm",
               title="Same state; three interventions")
    axs[0].legend(fontsize=8)
    xs=[r["depth_noise_std_m"]*1000 for r in backprojection]
    ys=[r["point_coordinate_rmse_m"]*1000 for r in backprojection]
    axs[1].plot(xs,ys,"o-",c="#315681")
    axs[1].set(xlabel="Depth noise std / mm",ylabel="3D coordinate RMSE / mm",
               title="Reconstruction noise: same pixels, same random draw")
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(OUT/"action_and_reconstruction.png",dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
