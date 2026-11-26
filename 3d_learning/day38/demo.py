"""D38: metric scale, projection, crop intrinsics and transform direction."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def project(x, r, t, k):
    camera=x@r.T+t
    homogeneous=camera@k.T
    return homogeneous[:,:2]/homogeneous[:,2:],camera


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop-left',type=float,default=160.)
    p.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day38')
    a=p.parse_args(); start=time.perf_counter(); out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(38)
    x=rng.uniform([-0.06,-0.04,-0.03],[.06,.04,.03],(120,3))
    angle=.35; c,s=np.cos(angle),np.sin(angle)
    r=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    t=np.array([.07,-.02,.70]); k=np.array([[600.,0,320],[0,610.,240],[0,0,1]])
    uv,camera=project(x,r,t,k)
    # Resize after crop: raw origin (left, top); scale is 1.5.
    top=100.; resize=1.5
    pixel_transform=np.array([[resize,0,-resize*a.crop_left],[0,resize,-resize*top],[0,0,1]])
    new_k=pixel_transform@k
    target=(uv-np.array([a.crop_left,top]))*resize
    corrected,_=project(x,r,t,new_k)
    wrong_k,_=project(x,r,t,k)
    uniform_scaled,scaled_camera=project(x*1000,r,t*1000,k)
    bad_mixed,_=project(x*1000,r,t,k)
    inverse_uv,inverse_camera=project(x,r.T,-r.T@t,k)
    back=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(k).T*camera[:,2:]
    raw_depth=1400.; depth_scale=.5
    depth_m=raw_depth*depth_scale/1000
    errors=dict(correct_crop=float(np.sqrt(np.mean(np.sum((corrected-target)**2,axis=1)))),
                wrong_K_after_crop=float(np.sqrt(np.mean(np.sum((wrong_k-target)**2,axis=1)))),
                uniform_scale_pixel=float(np.max(np.linalg.norm(uniform_scaled-uv,axis=1))),
                mixed_units_pixel=float(np.sqrt(np.mean(np.sum((bad_mixed-uv)**2,axis=1)))))
    assert errors['correct_crop']<1e-9 and errors['uniform_scale_pixel']<1e-9
    assert np.max(np.abs(back-camera))<1e-9
    result=dict(day=38,verified_on='2026-10-24',seed=38,device='CPU',python=platform.python_version(),
                numpy=np.__version__,matplotlib=matplotlib.__version__,training_steps=0,
                data='120 synthetic object points, calibrated ideal pinhole, no distortion',
                crop_left=a.crop_left,crop_top=top,resize=resize,K_after_crop=new_k.tolist(),
                errors_px=errors,correct_median_Z_m=float(np.median(camera[:,2])),
                mislabeled_scaled_median_Z_m=float(np.median(scaled_camera[:,2])),
                inverse_positive_depth_fraction=float(np.mean(inverse_camera[:,2]>0)),
                backprojection_max_error_m=float(np.max(np.abs(back-camera))),
                raw_depth=raw_depth,depth_scale_mm_per_raw=depth_scale,converted_depth_m=depth_m,
                hardware_interactions=0,elapsed_s=time.perf_counter()-start)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    np.savez(out/'arrays.npz',points_m=x,uv=uv,uv_crop=target,K=k,K_crop=new_k,R=r,t_m=t)
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    ax[0].scatter(uv[:,0],uv[:,1],s=12,label='original pixels'); ax[0].scatter(uniform_scaled[:,0],uniform_scaled[:,1],s=40,facecolors='none',edgecolors='#e49444',label='1000x X and t')
    ax[0].set_title('Projection cannot detect common scale'); ax[0].legend(fontsize=8)
    ax[1].scatter(target[:,0],target[:,1],s=12,label='crop labels')
    ax[1].scatter(wrong_k[:,0],wrong_k[:,1],s=12,label='old K'); ax[1].scatter(corrected[:,0],corrected[:,1],s=40,facecolors='none',edgecolors='#4c998b',label='A K')
    ax[1].set_title('Crop + resize must transform K'); ax[1].legend(fontsize=8)
    for q in ax: q.set_xlabel('u (pixel)'); q.set_ylabel('v (pixel)'); q.invert_yaxis(); q.set_aspect('equal')
    fig.savefig(out/'projection.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    ax[0].bar(['correct metric','1000x mislabeled'],[np.median(camera[:,2]),np.median(scaled_camera[:,2])]);ax[0].set_yscale('log');ax[0].set_ylabel('Median Z labeled as meters');ax[0].set_title('Use a metric depth check')
    ax[1].bar(['object to camera','its inverse'],[np.mean(camera[:,2]>0),np.mean(inverse_camera[:,2]>0)],color=['#4c998b','#e49444']);ax[1].set_ylim(0,1.1);ax[1].set_ylabel('Positive depth fraction');ax[1].set_title('Check transform direction')
    fig.savefig(out/'metric_checks.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
