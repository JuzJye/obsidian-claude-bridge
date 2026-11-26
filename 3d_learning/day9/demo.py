"""Day 9: epipolar geometry, DLT triangulation, planar IK. CPU only."""
from pathlib import Path
import json
import platform
import numpy as np

OUT=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day9"
L1,L2=.5,.4

def skew(t):
    x,y,z=t
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def project(P,X):
    h=np.c_[X,np.ones(len(X))] @ P.T
    return h[:,:2]/h[:,2:]

def triangulate(P1,P2,u1,u2):
    xyz=[]
    for a,b in zip(u1,u2):
        A=np.stack([a[0]*P1[2]-P1[0],a[1]*P1[2]-P1[1],
                    b[0]*P2[2]-P2[0],b[1]*P2[2]-P2[1]])
        _,_,vh=np.linalg.svd(A)
        h=vh[-1]
        if abs(h[3])<1e-12: raise ValueError("Point at infinity")
        xyz.append(h[:3]/h[3])
    return np.array(xyz)

def fk(q):
    q1,q2=q
    return np.array([L1*np.cos(q1)+L2*np.cos(q1+q2),
                     L1*np.sin(q1)+L2*np.sin(q1+q2)])

def analytical_ik(xy):
    x,y=xy; c=(x*x+y*y-L1*L1-L2*L2)/(2*L1*L2)
    if c < -1.-1e-12 or c > 1.+1e-12: return []
    angle=np.arccos(np.clip(c,-1.,1.))
    return [np.array([np.arctan2(y,x)-np.arctan2(L2*np.sin(q2),L1+L2*np.cos(q2)),q2])
            for q2 in [angle,-angle]]

def numerical_ik(target,q0):
    q=np.array(q0,dtype=float); history=[]
    for _ in range(100):
        e=target-fk(q); history.append(float(np.linalg.norm(e)))
        if history[-1]<1e-9: break
        # Finite differences here; Day10 derives the analytic Jacobian.
        eps=1e-6
        J=np.column_stack([(fk(q+np.eye(2)[i]*eps)-fk(q-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
        dq=np.linalg.pinv(J)@e
        dq*=min(1.,.2/max(np.linalg.norm(dq),1e-12))
        q+=dq
    return q,np.array(history)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(9)
    K=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    X=rng.uniform([-.3,-.2,1.],[.3,.2,4.],(200,3))
    P1=K@np.c_[np.eye(3),np.zeros(3)]
    # Same samples/noise at every baseline: baseline is the only variable.
    noise1=rng.normal(0.,.2,(len(X),2)); noise2=rng.normal(0.,.2,(len(X),2))
    baseline_results=[]; estimates=[]
    for baseline in [.03,.12,.24]:
        t=np.array([-baseline,0.,0.]); P2=K@np.c_[np.eye(3),t]
        u1,u2=project(P1,X),project(P2,X)
        E=skew(t); F=np.linalg.inv(K).T@E@np.linalg.inv(K)
        residual=np.einsum("ni,ij,nj->n",np.c_[u2,np.ones(len(X))],F,np.c_[u1,np.ones(len(X))])
        clean=triangulate(P1,P2,u1,u2)
        noisy=triangulate(P1,P2,u1+noise1,u2+noise2)
        baseline_results.append({"baseline_m":baseline,
            "max_epipolar_residual":float(np.max(np.abs(residual))),
            "clean_max_3d_error_m":float(np.max(np.linalg.norm(clean-X,axis=1))),
            "noisy_depth_rmse_m":float(np.sqrt(np.mean((noisy[:,2]-X[:,2])**2))),
            "positive_depth_fraction":float(np.mean(noisy[:,2]>0))})
        estimates.append(noisy)
        assert np.max(np.abs(clean-X))<1e-10
        assert np.max(np.abs(residual))<1e-10
    goal=np.array([.6,.2]); solutions=np.array(analytical_ik(goal))
    numerical=[]; histories=[]
    for q0 in [[0.,1.],[1.,-1.]]:
        q,h=numerical_ik(goal,q0); numerical.append(q); histories.append(h)
        assert h[-1]<1e-8
    assert len(analytical_ik([1.,0.]))==0
    assert np.max(np.linalg.norm(np.array([fk(q) for q in solutions])-goal,axis=1))<1e-12
    report={"environment":{"python":platform.python_version(),"numpy":np.__version__},
            "seed":9,"pixel_noise_std":.2,"triangulation":baseline_results,
            "goal_m":goal.tolist(),"analytical_ik_rad":solutions.tolist(),
            "numeric_ik_rad":[q.tolist() for q in numerical],
            "numeric_iterations":[len(h)-1 for h in histories],
            "numeric_final_error_m":[float(h[-1]) for h in histories],
            "unreachable_goal_m":[1.,0.],"unreachable_solutions":0,
            "scope":"known stereo extrinsics; no E/F estimation, collision checks or robot execution"}
    (OUT/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez(OUT/"arrays.npz",X=X,estimates=np.array(estimates),u1=u1,u2=u2,
             solutions=solutions,goal=goal,h0=histories[0],h1=histories[1])
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
