"""D55: RGB-D -> pose -> IK -> rate-limited control -> sensor feedback.
This is a kinematic teaching simulator with an explicit geometric attachment rule.
It has NO rigid-body contact dynamics and does NOT validate physical grasp success.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, json, platform, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H,W=80,96
K=np.array([[180.,0,48],[0,180.,40],[0,0,1]])
CAM=np.array([.4,0,1.])
L=np.array([.28,.26]); DT=.02; SENSOR_EVERY=5
OBJECT_HALF=np.array([.03,.018]); HEIGHT=.04
MAX_RATE=np.array([.7,.7,.12,1.2])  # rad/s, rad/s, m/s, rad/s
Q0=np.array([1.3,-1.6,.14,.3])
OBSTACLE=np.array([.14,-.16]); OBSTACLE_R=.025


def wrap(a): return (np.asarray(a)+np.pi)%(2*np.pi)-np.pi


def fk(q):
    elbow=L[0]*np.array([np.cos(q[0]),np.sin(q[0])])
    end=elbow+L[1]*np.array([np.cos(q[0]+q[1]),np.sin(q[0]+q[1])])
    return elbow,end,float(wrap(q[0]+q[1]+q[3]))


def segment_distance(p,a,b):
    v=b-a;s=np.clip(np.dot(p-a,v)/np.dot(v,v),0,1)
    return float(np.linalg.norm(p-a-s*v))


def clear(q):
    elbow,end,_=fk(q)
    return min(segment_distance(OBSTACLE,np.zeros(2),elbow),segment_distance(OBSTACLE,elbow,end))>OBSTACLE_R+.012


def ik(target,q):
    x,y,z,yaw=target
    c=(x*x+y*y-np.dot(L,L))/(2*np.prod(L))
    if abs(c)>1:return None,'unreachable'
    candidates=[]
    for q2 in [np.arccos(c),-np.arccos(c)]:
        q1=np.arctan2(y,x)-np.arctan2(L[1]*np.sin(q2),L[0]+L[1]*np.cos(q2))
        g=np.array([q1,q2,z,float(wrap(yaw-q1-q2))])
        if abs(q2)>2.8 or not (.01<=z<=.25):continue
        path=q+np.linspace(0,1,21)[:,None]*(g-q)
        if all(clear(p) for p in path):candidates.append(g)
    if not candidates:return None,'collision_or_limit'
    return min(candidates,key=lambda g:np.linalg.norm(g[[0,1,3]]-q[[0,1,3]])),None


def render(obj,noise,hidden=False):
    # Camera optical x -> base x; optical y -> -base y; optical z -> -base z.
    vv,uu=np.indices((H,W));ztop=obj[2]+HEIGHT/2;d=1-ztop
    worldx=CAM[0]+(uu-K[0,2])*d/K[0,0]
    worldy=-(vv-K[1,2])*d/K[1,1]
    dx=worldx-obj[0];dy=worldy-obj[1];c,s=np.cos(obj[3]),np.sin(obj[3])
    mask=(abs(c*dx+s*dy)<OBJECT_HALF[0])&(abs(-s*dx+c*dy)<OBJECT_HALF[1])
    if hidden:mask[:]=False
    rgb=np.full((H,W,3),60,dtype=np.uint8);rgb[mask]=[220,75,30]
    depth=np.ones((H,W));depth[mask]=d+noise[mask]
    return rgb,depth


def perceive(rgb,depth,bias_m):
    mask=(rgb[:,:,0]>160)&(rgb[:,:,1]<110)&(depth>0)&(depth<1.1)
    v,u=np.nonzero(mask)
    if len(u)<12:return None
    d=depth[v,u]+bias_m
    points=np.c_[CAM[0]+(u-K[0,2])*d/K[0,0],-(v-K[1,2])*d/K[1,1],1-d]
    center=points.mean(0);xy=points[:,:2]-center[:2]
    _,vectors=np.linalg.eigh(xy.T@xy/len(xy));axis=vectors[:,-1]
    yaw=(np.arctan2(axis[1],axis[0])+np.pi/2)%np.pi-np.pi/2
    minor=np.array([-np.sin(yaw),np.cos(yaw)])
    opening=float(np.ptp(xy@minor)+.012)
    if not .02<opening<.07:return None
    return np.array([center[0],center[1],center[2]+.01,yaw]),opening


def trial(case,policy,bias_m):
    # Each method sees identical initial state, shifts and sensor noise stream.
    rng=np.random.default_rng(5500+case)
    obj=np.array([rng.uniform(.37,.44),rng.uniform(.05,.11),.02,rng.uniform(-.4,.4)])
    mode=['static','object_shift','sensor_loss','unreachable'][case%4]
    if mode=='unreachable':obj[0]=.60
    initial=obj.copy();q=Q0.copy();reference=q.copy();stage='align'
    latest=None;fixed=None;last_seen=-np.inf;opening=0;attached=False;hold=0;reason='timeout'
    trace=[];images=None
    for step in range(300):
        t=step*DT
        if mode=='object_shift' and step==20:obj[:2]+=np.array([.035,-.025])
        if step%SENSOR_EVERY==0:
            noise=rng.normal(0,.0005,(H,W))
            rgb,depth=render(obj,noise,hidden=mode=='sensor_loss' and step>=20)
            if images is None:images=(rgb.copy(),depth.copy())
            estimate=perceive(rgb,depth,bias_m)
            if estimate is not None:
                latest,opening=estimate;last_seen=t
                if fixed is None:fixed=latest.copy()
        if latest is None or t-last_seen>.25:reason='stale_or_missing_observation';break
        goal=(latest if policy=='reobserve' else fixed).copy()
        if stage=='align':goal[2]=.14
        elif stage in ['lift','hold']:goal[2]=.16
        g,fail=ik(goal,q)
        if g is None:reason=fail;break
        reference+=np.clip(g-reference,-MAX_RATE*DT,MAX_RATE*DT)
        velocity=np.clip(8*(reference-q),-MAX_RATE,MAX_RATE)
        q+=.85*velocity*DT  # explicitly prescribed first-order velocity tracking
        elbow,ee,yaw=fk(q)
        if not clear(q):reason='sampled_collision';break
        if attached:obj=np.array([ee[0],ee[1],q[2]-.03,yaw])
        error=float(np.linalg.norm(ee-obj[:2]))
        trace.append([t,ee[0],ee[1],q[2],obj[0],obj[1],obj[2],error,t-last_seen])
        aligned=np.linalg.norm(ee-goal[:2])<.005 and abs(float(wrap(yaw-goal[3])))<.12
        if stage=='align' and aligned and abs(q[2]-.14)<.004:stage='descend'
        elif stage=='descend' and aligned and abs(q[2]-goal[2])<.004:
            # Privileged simulator truth is used ONLY for this geometric contact proxy.
            yaw_error=abs((yaw-obj[3]+np.pi/2)%np.pi-np.pi/2)
            attached=error<.008 and abs(q[2]-(obj[2]+HEIGHT/2+.01))<.006 and yaw_error<.26 and opening>=.036
            if not attached:reason='geometric_capture_miss';break
            stage='lift'
        elif stage=='lift' and q[2]>.154:stage='hold'
        elif stage=='hold':
            hold+=1
            if hold>=15:
                reason='proxy_success' if attached and obj[2]-initial[2]>.06 else 'lift_check_failed'
                break
    result=dict(case=case,condition=mode,policy=policy,outcome=reason,proxy_success=reason=='proxy_success',
                simulation_seconds=round(step*DT,3),sensor_frames=step//SENSOR_EVERY+1,
                final_true_xy_error_mm=float(trace[-1][7]*1000) if trace else None,
                physical_grasp_validated=False)
    return result,np.asarray(trace),images,initial


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--depth-bias-mm',type=float,default=0)
    ap.add_argument('--trials',type=int,default=40)
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day55')
    args=ap.parse_args()
    if args.trials<4 or args.trials%4:raise ValueError('trials must be a positive multiple of 4')
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    rows=[];traces={};snap=None
    for policy in ['frozen_pose','reobserve']:
        for case in range(args.trials):
            row,trace,images,initial=trial(case,policy,args.depth_bias_mm/1000);rows.append(row)
            if case==1:traces[policy]=trace;snap=images;start_object=initial
    summary={}
    for policy in ['frozen_pose','reobserve']:
        subset=[r for r in rows if r['policy']==policy]
        summary[policy]=dict(attempts=len(subset),proxy_successes=sum(r['proxy_success'] for r in subset),
                            outcomes=dict(Counter(r['outcome'] for r in subset)),
                            by_condition={c:dict(successes=sum(r['proxy_success'] for r in subset if r['condition']==c),
                            attempts=sum(r['condition']==c for r in subset)) for c in ['static','object_shift','sensor_loss','unreachable']})
    results=dict(seed_rule='5500 + case_id, reset separately for each method',python=platform.python_version(),
      numpy=np.__version__,matplotlib=matplotlib.__version__,depth_bias_mm=args.depth_bias_mm,
      control_dt_s=DT,camera_period_s=DT*SENSOR_EVERY,summary=summary,wall_seconds=time.perf_counter()-start,
      scene='analytic RGB-D pinhole camera, rectangular block, planar 2R plus vertical and wrist axes',
      success_definition='geometric capture AND synthetic attachment AND >60mm rise AND 15-step hold',
      limitations=['no MuJoCo/PyBullet','no contact/friction/inertia','no real RGB-D','no official pose model','not project physical-grasp acceptance'])
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    with (out/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez(out/'traces.npz',**traces)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].imshow(snap[0]);axes[0].set(title='Rendered RGB input',xlabel='u (px)',ylabel='v (px)')
    im=axes[1].imshow(snap[1],vmin=.95,vmax=1,cmap='viridis');axes[1].set(title='Optical depth (m)',xlabel='u (px)');fig.colorbar(im,ax=axes[1])
    for policy,trace in traces.items():axes[2].plot(trace[:,1],trace[:,2],label=policy)
    axes[2].scatter([start_object[0],start_object[0]+.035],[start_object[1],start_object[1]-.025],c=['black','red'],marker='x',s=70)
    axes[2].set(xlabel='Base x (m)',ylabel='Base y (m)',title='Moving-object trial: TCP trajectory');axes[2].axis('equal');axes[2].legend(fontsize=8)
    fig.savefig(out/'sensor_to_control.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    policies=list(summary)
    axes[0].bar(policies,[summary[p]['proxy_successes']/args.trials for p in policies],color=['#da6d3e','#2878a1'])
    for i,p in enumerate(policies):axes[0].text(i,.03,f"{summary[p]['proxy_successes']}/{args.trials}",ha='center')
    axes[0].set(ylim=(0,1),ylabel='Proxy success / ALL attempts',title='Complete attempt denominator')
    reasons=sorted(set(r['outcome'] for r in rows));bottom=np.zeros(2)
    for reason in reasons:
        values=np.array([summary[p]['outcomes'].get(reason,0) for p in policies]);axes[1].bar(policies,values,bottom=bottom,label=reason);bottom+=values
    axes[1].set(ylabel='Attempts',title='Outcome accounting');axes[1].legend(fontsize=6,loc='upper left')
    for policy,trace in traces.items():axes[2].plot(trace[:,0],trace[:,7]*1000,label=policy)
    axes[2].axvline(.4,ls='--',c='gray');axes[2].set(xlabel='Simulation time (s)',ylabel='True TCP-object xy error (mm)',title='Object shift at 0.4s');axes[2].legend(fontsize=8)
    fig.savefig(out/'closed_loop_results.png',dpi=160);plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
