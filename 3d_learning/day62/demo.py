"""D62: locally linear contact deviations, with/without recorded action. CPU."""
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate(rng, episodes=100, length=41):
    # States are deviations from a preload: dx [mm], dF [N], not absolute force.
    s = np.zeros((episodes, length, 2)); s[:, 0] = rng.normal(0, [.15, .2], (episodes, 2))
    u = rng.uniform(-2, 2, (episodes, length-1, 1))  # delta compression velocity [mm/s]
    for t in range(length-1):
        x = .95*s[:, t, 0] + .05*u[:, t, 0]
        f = .8*s[:, t, 1] + .6*x
        s[:, t+1] = np.c_[x, f] + rng.normal(0, [.001, .003], (episodes, 2))
    obs = s + rng.normal(0, [.002, .005], s.shape)
    return s, obs, u


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remove-action', action='store_true')
    p.add_argument('--out', type=Path, default=Path('_assets/3d_learning/day62/default'))
    a=p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); start=time.perf_counter()
    rng=np.random.default_rng(62); truth, obs, action=generate(rng)
    train, val, test=np.arange(60), np.arange(60,80), np.arange(80,100)
    scale=obs[train,:-1].reshape(-1,2).std(0)
    def features(s,u):
        return np.c_[s/scale, np.zeros(len(u)) if a.remove_action else u[:,0], np.ones(len(u))]
    x=features(obs[train,:-1].reshape(-1,2),action[train].reshape(-1,1))
    y=(obs[train,1:]-obs[train,:-1]).reshape(-1,2)/scale
    w=np.linalg.solve(x.T@x+1e-4*np.eye(4),x.T@y)
    def step(s,u): return s+(features(s,u)@w)*scale
    vy=step(obs[val,:-1].reshape(-1,2),action[val].reshape(-1,1))
    val_rmse=np.sqrt(np.mean((vy-obs[val,1:].reshape(-1,2))**2,0))
    starts=np.arange(0,31); pred=obs[test][:,starts].reshape(-1,2).copy()
    persist=pred.copy(); curves=[]; per_h={}; roll=[]
    for h in range(1,11):
        u=action[test][:,starts+h-1].reshape(-1,1)
        pred=step(pred,u); target=truth[test][:,starts+h].reshape(-1,2)
        rmse=np.sqrt(np.mean((pred-target)**2,0)); curves.append(rmse); roll.append(pred.copy())
        if h in (1,5,10):
            per_h[str(h)]={'windows':len(pred),'rmse_dx_mm':float(rmse[0]),'rmse_dF_N':float(rmse[1]),
                          'persistence_rmse_dx_mm':float(np.sqrt(np.mean((persist[:,0]-target[:,0])**2)))}
    # Same observation, opposing actions: no-action predictor must return same result.
    pair=step(np.array([[.2,.6],[.2,.6]]),np.array([[-2.],[2.]]))
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':62,'data':'synthetic local deviations',
            'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
            'remove_action':a.remove_action,'dt_s':.05,'action_unit':'mm/s','frame':'local normal compression axis',
            'train_episodes':60,'val_episodes':20,'test_episodes':20,'train_transitions':2400,
            'test_windows_per_horizon':620,'val_rmse_dx_mm_dF_N':val_rmse.tolist(),'free_rollout':per_h,
            'counterfactual_predictions_dx_mm_dF_N':pair.tolist(),'parameters':8,'robot_attempts':0,
            'environment_interactions':'4000 synthetic transitions, including held-out episodes',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':'Not calibrated contact dynamics; overlapping test windows are correlated; no confidence interval.'}
    np.savez_compressed(a.out/'results.npz',weights=w,state_scale=scale,truth=truth,observation=obs,action=action,
                        train_episodes=train,val_episodes=val,test_episodes=test,rollout=np.stack(roll))
    curves=np.array(curves); fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for j,label in enumerate(['Compression deviation RMSE [mm]','Force deviation RMSE [N]']):
        ax[j].plot(range(1,11),curves[:,j],'o-'); ax[j].set(xlabel='Free rollout horizon [steps]',ylabel=label)
        ax[j].grid(alpha=.25)
    fig.suptitle('Action omitted' if a.remove_action else 'Action-conditioned transition fit')
    fig.savefig(a.out/'rollout.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    target=truth[test][:,starts+10].reshape(-1,2)
    for j,unit in enumerate(['dx [mm]','dF [N]']):
        ax[j].scatter(target[:,j],pred[:,j],s=8,alpha=.35); lim=np.array([target[:,j].min(),target[:,j].max()])
        ax[j].plot(lim,lim,'k--'); ax[j].set(xlabel='True '+unit,ylabel='Predicted '+unit,title='10-step free rollout')
    fig.savefig(a.out/'prediction.png',dpi=160); plt.close(fig)
    report['wall_time_s']=time.perf_counter()-start
    (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
