"""D40: synthetic pose metrics; not a BOP benchmark submission."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def transform(x, r, t):
    return x @ r.T + t


def add(x, r, t, r_gt=np.eye(3), t_gt=np.zeros(3)):
    return float(np.linalg.norm(transform(x, r, t)-transform(x, r_gt, t_gt), axis=1).mean())


def adds(x, r, t, r_gt=np.eye(3), t_gt=np.zeros(3)):
    a, b = transform(x, r, t), transform(x, r_gt, t_gt)
    return float(np.linalg.norm(a[:, None]-b[None], axis=2).min(axis=1).mean())


def symmetry_add(x, r, t, symmetries):
    # One common symmetry for all vertices, unlike ADD-S nearest neighbours.
    return min(add(x, r, t, s, np.zeros(3)) for s in symmetries)


def diameter(x):
    return float(np.linalg.norm(x[:, None]-x[None], axis=2).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=.1, help='fraction of model diameter')
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day40')
    args = ap.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError('threshold must be a diameter fraction between 0 and 1')
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(40); start = time.perf_counter()
    angle = np.arange(32)*2*np.pi/32
    ring = np.array([[.04*np.cos(a), .04*np.sin(a), z] for a in angle for z in np.linspace(-.05,.05,5)])
    handle = np.array([[.075,y,z] for y in [-.005,.005] for z in [-.015,0,.015]])
    models = [ring, np.concatenate([ring,handle])]
    syms = [[rz(a) for a in [0,np.pi/2,np.pi,3*np.pi/2]], [np.eye(3)]]
    rows = []
    for i in range(80):
        obj = i//40; x = models[obj]
        present = i % 7 != 0
        a = rng.normal(0,.08) + (np.pi/2 if obj==0 and i%3==0 else 0)
        t = rng.normal(0,.003,3) + (np.array([.025,0,0]) if i%5==0 else 0)
        vals = [add(x,rz(a),t),adds(x,rz(a),t),symmetry_add(x,rz(a),t,syms[obj])]
        rows.append(dict(id=i,obj_id=obj,present=present,diameter_m=diameter(x),
                         add_m=vals[0] if present else None,
                         adds_m=vals[1] if present else None,
                         symmetry_add_m=vals[2] if present else None))
    thresholds = np.linspace(.001,.3,150)
    recalls = {}
    for key in ['add_m','adds_m','symmetry_add_m']:
        recalls[key]=[sum(r['present'] and r[key] < q*r['diameter_m'] for r in rows)/len(rows) for q in thresholds]
    success=sum(r['present'] and r['symmetry_add_m']<args.threshold*r['diameter_m'] for r in rows)
    found=sum(r['present'] for r in rows)
    special=dict(add_mm=1000*add(models[1],rz(np.pi/2),np.zeros(3)),
                 adds_mm=1000*adds(models[1],rz(np.pi/2),np.zeros(3)),
                 handle_displacement_mm=1000*np.linalg.norm(handle@rz(np.pi/2).T-handle,axis=1).mean())
    result=dict(seed=40,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                dataset='80 synthetic targets, 2 analytic models', units='m; figures mm',
                threshold_fraction=args.threshold,attempts=80,predictions=found,successes=success,
                recall_all=success/80,incorrect_conditional_recall=success/found,
                asymmetric_handle_counterexample=special,
                checks=dict(identity_add=add(ring,np.eye(3),np.zeros(3)),
                            translation_10mm=add(ring,np.eye(3),np.array([.01,0,0])),
                            c4_rotation_adds=adds(ring,rz(np.pi/2),np.zeros(3))),
                timing_seconds=time.perf_counter()-start,
                scope='No trained model, real data, BOP VSD/MSSD/MSPD evaluation or robot execution')
    assert abs(result['checks']['translation_10mm']-.01)<1e-12
    assert result['checks']['c4_rotation_adds']<1e-12
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'per_target.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for key,y in recalls.items(): axes[0].plot(thresholds,y,label=key.removesuffix('_m'))
    axes[0].axvline(args.threshold,color='gray',ls='--'); axes[0].set(xlabel='Threshold / diameter',ylabel='Recall (all 80 targets)',ylim=(0,1)); axes[0].legend()
    axes[1].bar(['All targets','Predicted only'],[success/80,success/found],color=['#2878a1','#da6d3e'])
    for k,v in enumerate([success/80,success/found]): axes[1].text(k,v+.015,f'{success}/{[80,found][k]} = {v:.3f}',ha='center')
    axes[1].set(ylim=(0,1),ylabel='Success fraction',title='Changing the denominator changes the claim')
    fig.savefig(out/'recall_protocol.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    axes[0].scatter(models[1][:,0]*1000,models[1][:,1]*1000,s=9,label='GT')
    rotated=models[1]@rz(np.pi/2).T
    axes[0].scatter(rotated[:,0]*1000,rotated[:,1]*1000,s=9,label='90 degree estimate'); axes[0].axis('equal')
    axes[0].set(xlabel='x (mm)',ylabel='y (mm)',title='Asymmetric handle on a mostly symmetric body');axes[0].legend()
    axes[1].bar(['ADD','ADD-S','Handle'],list(special.values()),color=['#2878a1','#48a086','#da6d3e'])
    axes[1].set(ylabel='Displacement (mm)',title='Global surface average vs task feature')
    fig.savefig(out/'symmetry_boundary.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
