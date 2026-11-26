"""D39: oracle-correspondence pose baselines; NOT NOCS/EPro-PnP inference."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def similarity(x,y):
    mx,my=x.mean(0),y.mean(0)
    xc,yc=x-mx,y-my
    u,s,vt=np.linalg.svd(yc.T@xc/len(x))
    d=np.eye(3);d[2,2]=np.linalg.det(u@vt)
    r=u@d@vt
    scale=np.sum(s*np.diag(d))/np.mean(np.sum(xc*xc,axis=1))
    return scale,r,my-scale*r@mx


def rigid(x,y):
    mx,my=x.mean(0),y.mean(0)
    u,_,vt=np.linalg.svd((y-my).T@(x-mx))
    d=np.eye(3);d[2,2]=np.linalg.det(u@vt)
    r=u@d@vt
    return r,my-r@mx


def rotation_error(a,b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outlier-fraction',type=float,default=.20)
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day39')
    a=ap.parse_args()
    if not 0<=a.outlier_fraction<.7:raise ValueError('use outlier fraction in [0,.7)')
    start=time.perf_counter();out=a.out_dir;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(39);cv2.setRNGSeed(39);cv2.setNumThreads(1)
    n=100
    canonical=rng.uniform(-.5,.5,(n,3));scale=.18;x=canonical*scale
    rtrue,_=cv2.Rodrigues(np.array([.25,-.15,.08]));ttrue=np.array([.03,-.02,.75])
    k=np.array([[600.,0,320],[0,600.,240],[0,0,1]])
    y=x@rtrue.T+ttrue;h=y@k.T;uv=h[:,:2]/h[:,2:]
    noisy_uv=uv+rng.normal(0,.6,uv.shape)
    noisy_y=y+rng.normal(0,.001,y.shape)
    count=int(n*a.outlier_fraction);bad=np.arange(count)
    corruption_rng=np.random.default_rng(390)
    noisy_uv[bad]+=corruption_rng.uniform(-80,80,(n,2))[bad]
    noisy_y[bad]+=corruption_rng.uniform(-.08,.08,(n,3))[bad]
    methods=[]
    def report(name,r,t,sc=1.,inliers=None):
        predicted=sc*x@r.T+t
        methods.append(dict(method=name,translation_error_mm=float(np.linalg.norm(t-ttrue)*1000),
                            rotation_error_deg=rotation_error(r,rtrue),
                            mean_point_error_mm=float(np.mean(np.linalg.norm(predicted-y,axis=1))*1000),
                            scale_multiplier=float(sc),inliers=inliers))
    ok,rv,tv=cv2.solvePnP(x,noisy_uv,k,None,flags=cv2.SOLVEPNP_EPNP)
    if not ok:raise RuntimeError('EPnP failed')
    r,_=cv2.Rodrigues(rv);report('EPnP all',r,tv.ravel())
    ok,rv,tv,inside=cv2.solvePnPRansac(x,noisy_uv,k,None,iterationsCount=500,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inside is None or len(inside)<6:raise RuntimeError('RANSAC failed')
    idx=inside.ravel();rv,tv=cv2.solvePnPRefineLM(x[idx],noisy_uv[idx],k,None,rv,tv)
    r,_=cv2.Rodrigues(rv);report('RANSAC PnP',r,tv.ravel(),inliers=len(idx))
    r,t=rigid(x,noisy_y);report('rigid SVD all',r,t)
    best=np.zeros(n,bool); sampling_rng=np.random.default_rng(391)
    for _ in range(300):
        ix=sampling_rng.choice(n,3,replace=False)
        r,t=rigid(x[ix],noisy_y[ix])
        good=np.linalg.norm(x@r.T+t-noisy_y,axis=1)<.005
        if good.sum()>best.sum():best=good
    r,t=rigid(x[best],noisy_y[best]);report('RANSAC 3D',r,t,inliers=int(best.sum()))
    # Similarity alignment uses only clean 3D correspondences to isolate scale recovery.
    sc,r,t=similarity(canonical,y+np.random.default_rng(392).normal(0,.001,y.shape))
    sim_error=float(abs(sc-scale)*1000)
    # A 1D translation posterior slice: analytic Gaussian, not SE(3) EPro-PnP.
    offsets=np.linspace(-.01,.01,401);sigmas=[.001,.004];density=[]
    for sigma in sigmas:
        p=np.exp(-.5*(offsets/sigma)**2);p/=np.trapz(p,offsets);density.append(p)
    result=dict(day=39,verified_on='2026-10-24',seed=39,device='CPU',python=platform.python_version(),
                numpy=np.__version__,opencv=cv2.__version__,matplotlib=matplotlib.__version__,
                data='100 synthetic 3D points with oracle correspondences and injected corruptions',
                outlier_fraction=a.outlier_fraction,noise_image_px=.6,noise_depth_xyz_m=.001,
                comparisons=methods,similarity_true_scale_m=scale,similarity_scale_m=float(sc),
                similarity_scale_error_mm=sim_error,similarity_rotation_deg=rotation_error(r,rtrue),
                similarity_translation_error_mm=float(np.linalg.norm(t-ttrue)*1000),
                training_steps=0,hardware_interactions=0,elapsed_s=time.perf_counter()-start)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    names=[m['method'] for m in methods]
    ax[0].bar(names,[m['mean_point_error_mm'] for m in methods],color=['#e49444','#4c998b','#e49444','#4c998b'])
    ax[0].set_ylabel('Mean 3D point error (mm)');ax[0].tick_params(axis='x',rotation=25);ax[0].set_title(f'Oracle correspondences, {a.outlier_fraction:.0%} corruption')
    ax[1].scatter(uv[:,0],uv[:,1],s=10,label='clean pixels');ax[1].scatter(noisy_uv[bad,0],noisy_uv[bad,1],s=20,label='corrupted pixels')
    ax[1].invert_yaxis();ax[1].set_xlabel('u (px)');ax[1].set_ylabel('v (px)');ax[1].legend();ax[1].set_title('Injected mismatch is visible')
    fig.savefig(out/'baselines.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.6),constrained_layout=True)
    for sigma,p in zip(sigmas,density):ax.plot(offsets*1000,p,label=f'sigma = {sigma*1000:.0f} mm')
    ax.set_xlabel('Translation offset (mm)');ax.set_ylabel('Normalized density (1/m)');ax.legend();ax.set_title('Same MAP, different uncertainty: 1D teaching slice')
    fig.savefig(out/'uncertainty.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
