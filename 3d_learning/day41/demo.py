"""D41: timestamp contract and action-conditioned transitions, CPU synthetic data."""
from pathlib import Path
import argparse,json,platform,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--latency-ms',type=float,default=80)
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day41')
    args=ap.parse_args()
    if args.latency_ms<0: raise ValueError('negative latency')
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(41); start=time.perf_counter()
    dt=.05
    t=np.arange(201)*dt
    u=.12*np.sin(2*np.pi*t/.6)+.04*np.sin(2*np.pi*t/1.9)
    x=np.r_[.3,.3+np.cumsum(u[:-1]*dt)]
    exposure=np.arange(300)/30+rng.uniform(-.002,.002,300)
    exposure=np.clip(exposure,0,t[-1]); exposure.sort()
    arrival=exposure+args.latency_ms/1000+rng.uniform(0,.01,len(exposure))
    observed=np.interp(exposure,t,x)
    ticks=np.arange(5,191); q=t[ticks]
    # Offline labels may interpolate exposure-bracketing observations.
    correct=np.interp(q,exposure,observed)
    # Wrong: assign the old measurement to the time it arrives.
    wrong=np.interp(q,arrival,observed)
    pairs={name:np.diff(a) for name,a in [('oracle',x[ticks]),('exposure',correct),('arrival',wrong)]}
    feature=u[ticks[:-1]]*dt
    train=np.arange(len(feature))<120
    metrics={}; gains={}
    for name,delta in pairs.items():
        g=float(np.dot(feature[train],delta[train])/np.dot(feature[train],feature[train]))
        gains[name]=g
        metrics[name]=dict(gain=g,heldout_label_rmse_mm=1000*rms(g*feature[~train]-delta[~train]),
                           heldout_true_step_rmse_mm=1000*rms(g*feature[~train]-pairs['oracle'][~train]))
    # Online only use an observation that has actually arrived.
    latest=np.searchsorted(arrival,q,side='right')-1
    valid=latest>=0
    assert np.all(arrival[latest[valid]]<=q[valid])
    ages=q[valid]-exposure[latest[valid]]
    stale_error=observed[latest[valid]]-x[ticks[valid]]
    future_count=sum(np.searchsorted(exposure,s,side='right')<len(exposure) for s in q)
    # One demonstration of a reset boundary: no transition crosses episode IDs.
    episode=np.zeros(len(q),int);episode[100:]=1
    reset_x=x[ticks].copy();reset_x[100:]+=.3-reset_x[100]
    same_episode=episode[1:]==episode[:-1]
    reset_jump=abs(np.diff(reset_x)[~same_episode][0])
    # Same true displacements but a metadata period of 100ms instead of 50ms.
    wrong_dt_gain=float(np.dot(2*feature[train],pairs['oracle'][train])/np.dot(2*feature[train],2*feature[train]))
    rollout={}
    for name,g in gains.items():
        rollout[name]={}
        for horizon in [1,5,10]:
            starts=np.arange(130,190-horizon)
            pred=np.array([x[k]+g*np.sum(u[k:k+horizon])*dt for k in starts])
            rollout[name][str(horizon)]=1000*rms(pred-x[starts+horizon])
    result=dict(seed=41,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                unit='m, m/s, s',action='applied world-x velocity held on [t,t+0.05)',
                state='synthetic noiseless position; camera measurements linearly interpolated',
                latency_base_ms=args.latency_ms,dt_seconds=dt,train_transitions=120,
                test_transitions=int((~train).sum()),metrics=metrics,rollout_true_rmse_mm=rollout,
                online_mean_age_ms=1000*float(ages.mean()),online_stale_position_rmse_mm=1000*rms(stale_error),
                offline_future_brackets=int(future_count),reset_pairs_rejected=int((~same_episode).sum()),
                reset_jump_mm=1000*float(reset_jump),wrong_period_gain=wrong_dt_gain,
                all_online_observations_already_arrived=True,elapsed_seconds=time.perf_counter()-start,
                scope='Known synthetic dynamics; no trained robot, ROS stream, VLA or world-model benchmark')
    assert abs(gains['oracle']-1)<1e-12 and abs(wrong_dt_gain-.5)<1e-12
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez(out/'arrays.npz',t=q,truth=x[ticks],exposure_aligned=correct,arrival_aligned=wrong,
             exposure=exposure,arrival=arrival,observed=observed,applied_velocity=u[ticks])
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    s=q<2
    for name,a in [('truth',x[ticks]),('exposure labels',correct),('arrival labels',wrong)]: axes[0].plot(q[s],1000*a[s],label=name)
    axes[0].set(xlabel='Physical time (s)',ylabel='World x (mm)',title='Transport latency changes sample meaning');axes[0].legend()
    axes[1].hist(ages*1000,bins=16,color='#2878a1');axes[1].set(xlabel='Online observation age (ms)',ylabel='Count',title='Causal does not mean current')
    fig.savefig(out/'timestamp_alignment.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    axes[0].bar(list(gains),list(gains.values()),color=['#48a086','#2878a1','#da6d3e']);axes[0].axhline(1,ls='--',c='gray')
    axes[0].set(ylabel='Fitted velocity gain',title='Same system, different labels')
    for name,r in rollout.items():axes[1].plot([1,5,10],list(r.values()),'o-',label=name)
    axes[1].set(xlabel='Prediction horizon (steps)',ylabel='True rollout RMSE (mm)');axes[1].legend()
    fig.savefig(out/'transition_error.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
