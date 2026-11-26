"""D61 one-dimensional quasistatic contact hybrid controller; not robot physics."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def simulate(seed, delay_ms):
    rng=np.random.default_rng(seed); dt=.005; k=1800.; gain=.012; vmax=.02; fmax=8.
    n_delay=int(round(delay_ms/1000/dt)); x=-.004; state=0; onset=0; target=3.; slip_total=0.; rows=[]; history=[]
    # states: approach, hold, recover, release, done, fault
    for i in range(700):
        now=i*dt; force=max(0.,k*x); history.append(max(0.,force+rng.normal(0,.04)))
        idx=max(0,i-n_delay); measured=history[idx]; age=(i-idx)*dt
        load=0. if now<1. else (1.8 if now<2.4 else 0.)
        slip=max(0.,load-.4*force)*.003 if state in (1,2) else 0.
        slip_total+=slip*dt
        if age>.05 or measured>fmax: state=5
        if state==0:
            onset=onset+1 if measured>.6 else 0
            if onset>=3: state=1
        if state==1 and slip>.00015: state=2; target=5.2
        if state==2 and now>1.2 and abs(measured-target)<.15: state=1
        if now>=2.4 and state in (0,1,2): state=3
        if state==3 and x<=-.002: state=4
        if state==0: v=.008
        elif state in (1,2): v=float(np.clip(gain*(target-measured),-vmax,vmax))
        elif state==3: v=-.01
        else: v=0.
        rows.append([now,x,force,measured,target,v,state,slip_total,age]); x+=v*dt
    arr=np.array(rows); success=state==4 and arr[:,2].max()<fmax and slip_total<.001
    return arr,{'seed':seed,'terminal_state':['approach','hold','recover','release','done','fault'][state],
      'success':bool(success),'peak_force_N':float(arr[:,2].max()),'slip_mm':1000*slip_total,
      'max_speed_m_s':float(np.abs(arr[:,5]).max()),'max_age_ms':float(arr[:,8].max()*1000)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--delay-ms',type=float,default=0.)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day61/default')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    cases=[simulate(s,args.delay_ms) for s in range(61,81)]; arr=cases[0][0]; reports=[c[1] for c in cases]
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axs[0].plot(arr[:,0],arr[:,2],label='True force'); axs[0].plot(arr[:,0],arr[:,3],alpha=.6,label='Delayed measured force'); axs[0].plot(arr[:,0],arr[:,4],ls='--',label='Hold target'); axs[0].axhline(8,color='r',ls=':',label='Limit'); axs[0].set(ylabel='Force (N)',title='Synthetic normal contact'); axs[0].legend(ncol=2)
    axs[1].step(arr[:,0],arr[:,6],where='post'); axs[1].set(xlabel='Time (s)',ylabel='State',yticks=range(6),yticklabels=['approach','hold','recover','release','done','fault']); fig.tight_layout(); fig.savefig(args.out/'contact.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4)); axs[0].plot(arr[:,0],1000*arr[:,5]); axs[0].set(xlabel='Time (s)',ylabel='Normal velocity (mm/s)',title='Rate-limited command')
    delays=[0,10,30,80]; sweeps=[]
    for d in delays:
        rr=[simulate(s,d)[1] for s in range(61,81)]; sweeps.append(sum(r['success'] for r in rr))
    axs[1].bar([str(d) for d in delays],sweeps); axs[1].set(xlabel='Observation delay (ms)',ylabel='Successful / 20',ylim=(0,21),title='Same seeds; stale-data gate active'); fig.tight_layout(); fig.savefig(args.out/'feedback.png',dpi=160); plt.close(fig)
    report={'python':platform.python_version(),'numpy':np.__version__,'delay_ms':args.delay_ms,'dt_s':.005,'stiffness_N_m':1800.,'gain_m_per_N_s':.012,
      'force_limit_N':8.,'speed_limit_m_s':.02,'stale_limit_ms':50.,'successes':sum(r['success'] for r in reports),'attempts':20,
      'trials':reports,'sweep_delays_ms':delays,'sweep_successes':sweeps,'runtime_s':time.perf_counter()-start,
      'data':'analytic spring contact and algebraic slip proxy; no inertial/contact simulation or hardware',
      'robot_trials':0,'learning_status':'not_assessed'}
    np.savez(args.out/'trace.npz',trace=arr); (args.out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
