"""Day 10: PnP, block matching and differential kinematics. CPU only."""
from pathlib import Path
import json
import platform
import numpy as np
import cv2

OUT=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day10"
L1,L2=.5,.4

def fk(q):
    a,b=q
    return np.array([L1*np.cos(a)+L2*np.cos(a+b),L1*np.sin(a)+L2*np.sin(a+b)])

def jacobian(q):
    a,b=q
    return np.array([[-L1*np.sin(a)-L2*np.sin(a+b),-L2*np.sin(a+b)],
                     [ L1*np.cos(a)+L2*np.cos(a+b), L2*np.cos(a+b)]])

def dls(J,error,damping):
    return J.T@np.linalg.solve(J@J.T+damping*damping*np.eye(J.shape[0]),error)

def patch_cost(left,right,u=80,v=48,r=3,max_d=24):
    ref=left[v-r:v+r+1,u-r:u+r+1].astype(float)
    return np.array([np.mean((ref-right[v-r:v+r+1,u-d-r:u-d+r+1].astype(float))**2)
                     for d in range(max_d)])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(10)
    K=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    X=rng.uniform([-.15,-.1,-.08],[.15,.1,.08],(60,3))
    rtrue=np.array([.15,-.20,.05]); ttrue=np.array([.04,-.02,.9])
    uv=cv2.projectPoints(X,rtrue,ttrue,K,None)[0].reshape(-1,2)
    unit_noise=rng.normal(size=uv.shape)
    cases=[]; reprojects=[]
    for sigma in [0.,.5,2.]:
        obs=uv+sigma*unit_noise
        ok,r,t=cv2.solvePnP(X,obs,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
        assert ok
        pred=cv2.projectPoints(X,r,t,K,None)[0].reshape(-1,2)
        R,_=cv2.Rodrigues(r); Rtrue,_=cv2.Rodrigues(rtrue)
        angle=np.arccos(np.clip((np.trace(R@Rtrue.T)-1)/2,-1,1))
        cases.append({"pixel_noise_std":sigma,"translation_error_m":float(np.linalg.norm(t.ravel()-ttrue)),
                      "rotation_error_deg":float(np.rad2deg(angle)),
                      "reprojection_rmse_px":float(np.sqrt(np.mean(np.sum((pred-obs)**2,axis=1))))})
        reprojects.append(pred)
    # Frontoparallel synthetic plane: one disparity, ideal rectification/photometry.
    left=rng.integers(0,256,(96,160),dtype=np.uint8)
    right=rng.integers(0,256,left.shape,dtype=np.uint8)
    disparity_true=12;right[:,:-disparity_true]=left[:,disparity_true:]
    flat=np.full_like(left,127)
    textured_cost=patch_cost(left,right); flat_cost=patch_cost(flat,flat)
    stereo=cv2.StereoBM_create(numDisparities=16,blockSize=9)
    disparity=stereo.compute(left,right).astype(np.float32)/16.
    roi=np.zeros_like(disparity,dtype=bool);roi[8:-8,24:-16]=True
    valid=roi&(disparity>0)
    depth=600.*.12/disparity[valid]
    q=np.array([.4,1.]);J=jacobian(q);eps=1e-6
    finite=np.column_stack([(fk(q+np.eye(2)[i]*eps)-fk(q-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    local=[]
    direction=np.array([.3,-.2])
    for scale in [1.,.1,.01]:
        dq=scale*direction
        exact=fk(q+dq)-fk(q); approx=J@dq
        local.append({"scale":scale,"delta_q_rad":dq.tolist(),"exact_delta_m":exact.tolist(),
                      "linear_delta_m":approx.tolist(),"linearization_error_m":float(np.linalg.norm(exact-approx))})
    near=np.array([0.,.01]);Jn=jacobian(near);error=np.array([-.01,0.])
    pinv_step=np.linalg.pinv(Jn)@error
    dls_step=dls(Jn,error,.05)
    singular=jacobian([0.,0.]); singular_values=np.linalg.svd(Jn,compute_uv=False)
    assert np.max(np.abs(J-finite))<1e-8
    assert cases[0]["translation_error_m"]<1e-8
    assert np.argmin(textured_cost)==disparity_true
    assert np.ptp(flat_cost)==0
    assert np.linalg.norm(dls_step)<np.linalg.norm(pinv_step)
    assert np.linalg.matrix_rank(singular)==1
    assert local[2]["linearization_error_m"]<local[1]["linearization_error_m"]<local[0]["linearization_error_m"]
    report={"environment":{"python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__},
            "seed":10,"pnp":cases,"stereo":{"true_disparity_px":12,"true_depth_m":6.,
            "valid_fraction_in_roi":float(valid.sum()/roi.sum()),"depth_median_m":float(np.median(depth)),
            "disparity_mae_px":float(np.mean(abs(disparity[valid]-12))),
            "textured_best_disparity":int(np.argmin(textured_cost)),"flat_equal_minima":int(np.sum(flat_cost==flat_cost.min()))},
            "jacobian":J.tolist(),"finite_difference_max_error":float(np.max(np.abs(J-finite))),
            "local_linearization":local,"near_singular_values":singular_values.tolist(),
            "near_singular_condition":float(singular_values[0]/singular_values[-1]),
            "desired_delta_m":error.tolist(),"pinv_delta_q_rad":pinv_step.tolist(),"dls_delta_q_rad":dls_step.tolist(),
            "pinv_nonlinear_residual_m":float(np.linalg.norm(fk(near+pinv_step)-fk(near)-error)),
            "dls_nonlinear_residual_m":float(np.linalg.norm(fk(near+dls_step)-fk(near)-error)),
            "scope":"synthetic non-planar PnP, ideal textured plane, single-step DLS; no full IK or robot execution"}
    (OUT/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez(OUT/"arrays.npz",left=left,right=right,disparity=disparity,valid=valid,
             textured_cost=textured_cost,flat_cost=flat_cost,uv=uv,noisy_uv=uv+2*unit_noise,
             reproject=reprojects[-1],q=q,J=J,near=near,Jn=Jn,pinv=pinv_step,dls=dls_step)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
