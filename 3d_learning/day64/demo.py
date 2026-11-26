"""D64: early, late and reliability-gated fusion; identical split. CPU."""
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ridge(x,y): return np.linalg.solve(x.T@x+1e-4*np.eye(x.shape[1]),x.T@y)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--touch-noise-mult',type=float,default=1.)
    ap.add_argument('--out',type=Path,default=Path('_assets/3d_learning/day64/default'))
    args=ap.parse_args()
    if not 1<=args.touch_noise_mult<=10: ap.error('noise multiplier must be in [1,10]')
    args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); rng=np.random.default_rng(64)
    n=4000; y=rng.normal(size=n); qv=rng.integers(0,2,n); qt=1-qv
    sv=np.where(qv==1,.1,.8); st=np.where(qt==1,.1,.8)
    nv=rng.normal(size=n); nt=rng.normal(size=n); v=y+sv*nv; t=y+st*nt
    train=np.arange(2400); val=np.arange(2400,3200); test=np.arange(3200,4000)
    # Same labelled training budget. Masks augment EARLY training without new labels.
    mv=np.ones(n); mt=np.ones(n); drop=rng.uniform(size=2400)
    mv[:2400]=(drop>=.2); mt[:2400]=~((drop>=.2)&(drop<.4))
    def early(vv,tt,vm,tm,q1,q2): return np.c_[vm*vv,tm*tt,vm,tm,q1,q2,np.ones(len(vv))]
    ew=ridge(early(v[train],t[train],mv[train],mt[train],qv[train],qt[train]),y[train])
    # Single-modality calibrated heads, shared by late and gated variants.
    wv=ridge(np.c_[v[train],np.ones(len(train))],y[train])
    wt=ridge(np.c_[t[train],np.ones(len(train))],y[train])
    pv=wv[0]*v+wv[1]; pt=wt[0]*t+wt[1]
    # One closed-form validation fit for the fixed late-fusion weight.
    d=pv[val]-pt[val]; alpha=float(np.clip(d@(y[val]-pt[val])/(d@d),0,1))
    varv=np.array([np.mean((pv[train][qv[train]==q]-y[train][qv[train]==q])**2) for q in [0,1]])
    vart=np.array([np.mean((pt[train][qt[train]==q]-y[train][qt[train]==q])**2) for q in [0,1]])
    outputs={}; metrics={}
    for name,missing in [('both_present',False),('touch_missing',True)]:
        vv=v[test]; tt=y[test]+st[test]*nt[test]*args.touch_noise_mult
        vm=np.ones(800); tm=np.zeros(800) if missing else np.ones(800)
        p1=wv[0]*vv+wv[1]; p2=wt[0]*tt+wt[1]
        l1=alpha*vm; l2=(1-alpha)*tm
        g1=vm/varv[qv[test]]; g2=tm/vart[qt[test]]
        predictions={'vision_only':p1,'early_linear':early(vv,tt,vm,tm,qv[test],qt[test])@ew,
                     'late_fixed':(l1*p1+l2*p2)/(l1+l2),'gated_quality':(g1*p1+g2*p2)/(g1+g2)}
        outputs[name]=predictions
        metrics[name]={k:{'rmse_mm':float(np.sqrt(np.mean((pred-y[test])**2))),
                          'mae_mm':float(np.mean(abs(pred-y[test]))),'denominator':800}
                       for k,pred in predictions.items()}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':64,'python':platform.python_version(),
            'numpy':np.__version__,'matplotlib':matplotlib.__version__,'data':'synthetic scalar local position in mm',
            'train_val_test_independent_episodes':[2400,800,800],'test_touch_noise_multiplier':args.touch_noise_mult,
            'late_visual_weight':alpha,'estimated_visual_residual_variance_mm2':varv.tolist(),
            'estimated_touch_residual_variance_mm2':vart.tolist(),'metrics':metrics,'robot_attempts':0,
            'parameters':{'early':7,'shared_unimodal_heads':4,'late_weight':1,'gate_variances':4},
            'limits':'Gate uses declared quality categories, not oracle test errors; test noise change leaves those categories stale. '
                     'Different inductive biases and parameter counts; equal data budget is not equal capacity.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    np.savez_compressed(args.out/'results.npz',truth_test=y[test],quality_visual=qv[test],quality_touch=qt[test],
                        **{scenario+'_'+name:pred for scenario,ps in outputs.items() for name,pred in ps.items()})
    methods=list(outputs['both_present']); fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for i,scenario in enumerate(outputs):
        ax.bar(np.arange(4)+(i-.5)*.35,[metrics[scenario][m]['rmse_mm'] for m in methods],.35,label=scenario)
    ax.set(xticks=np.arange(4),xticklabels=methods,ylabel='RMSE [mm]',title=f'Test touch noise multiplier = {args.touch_noise_mult:g}')
    ax.legend(); fig.savefig(args.out/'comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for j,(scenario,ps) in enumerate(outputs.items()):
        for method in ['early_linear','gated_quality']:
            ax[j].plot(y[test][:50]-ps[method][:50],label=method)
        ax[j].axhline(0,color='black',lw=.5); ax[j].set(xlabel='Held-out episode ID (first 50)',ylabel='Signed error [mm]',title=scenario)
        ax[j].legend()
    fig.savefig(args.out/'errors.png',dpi=160); plt.close(fig)
    report['wall_time_s']=time.perf_counter()-start
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
