"""D43: two-scale geometric features, paired density/noise controls, no deep backbone."""
from pathlib import Path
import argparse,json,platform,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def clouds(seed,m,n=256,noise=.003):
    # Always draw 256 points, then take a subset: sparse is paired with dense.
    rng=np.random.default_rng(seed);labels=rng.integers(0,4,m)
    local=np.where(labels%2,1,-1);global_sign=np.where(labels//2,1,-1)
    angle=rng.uniform(0,2*np.pi,(m,256));radius=.12*np.sqrt(rng.uniform(0,1,(m,256)))
    x=radius*np.cos(angle);y=radius*np.sin(angle)
    epsilon=rng.normal(0,1,(m,256))
    z=.15*global_sign[:,None]*x+.02*local[:,None]*np.exp(-radius**2/(2*.008**2))+noise*epsilon
    return np.stack([x,y,z],axis=-1)[:,:n],labels


def features(points,radii):
    n_batch=points.shape[0];output=[];counts=[]
    for radius in radii:
        branch=[];sizes=[]
        for p in points:
            mask=np.linalg.norm(p[:,:2],axis=1)<radius
            sizes.append(int(mask.sum()))
            if mask.sum()<3:
                branch.append([0.,0.]);continue
            # Local plane z = intercept + bx*x + by*y; only intercept,bx retained.
            A=np.c_[np.ones(mask.sum()),p[mask,:2]]
            coef=np.linalg.lstsq(A,p[mask,2],rcond=None)[0]
            branch.append(coef[:2])
        output.append(np.asarray(branch));counts.append(sizes)
    return np.concatenate(output,axis=1),np.array(counts)


def fit(f,y):
    mean=f.mean(0);std=f.std(0)+1e-8
    A=np.c_[(f-mean)/std,np.ones(len(f))]
    W=np.linalg.solve(A.T@A+1e-3*np.eye(A.shape[1]),A.T@np.eye(4)[y])
    return mean,std,W


def predict(f,model):
    mean,std,W=model
    return np.argmax(np.c_[(f-mean)/std,np.ones(len(f))]@W,axis=1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--small-radius-mm',type=float,default=25)
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day43')
    args=ap.parse_args()
    if not 5<=args.small_radius_mm<100:raise ValueError('small radius must be in [5,100) mm')
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    radii=[args.small_radius_mm/1000,.1]
    train,y_train=clouds(43,1024);F,train_counts=features(train,radii)
    specs={'small':[0,1],'large':[2,3],'multiscale':[0,1,2,3],'duplicate_small':[0,1,0,1]}
    models={name:fit(F[:,ids],y_train) for name,ids in specs.items()}
    settings={'dense':(256,.003),'sparse':(64,.003),'noisy':(256,.006)}
    scores={};group_stats={};test_features={};test_labels=None
    for condition,(n,noise) in settings.items():
        points,y=clouds(430,512,n,noise);f,counts=features(points,radii)
        test_features[condition]=f;test_labels=y
        group_stats[condition]=dict(mean_neighbors=counts.mean(axis=1).tolist(),
            small_insufficient_fraction=float(np.mean(counts[0]<3)))
        scores[condition]={}
        for name,ids in specs.items():
            prediction=predict(f[:,ids],models[name]);correct=int(np.sum(prediction==y))
            scores[condition][name]=dict(correct=correct,total=512,accuracy=correct/512)
    # Numerical point-order invariance: same local fits after permuting inputs.
    perm=np.random.default_rng(431).permutation(256)
    permF,_=features(train[:8,perm],radii)
    permutation_error=float(np.max(abs(permF-F[:8])))
    assert permutation_error<1e-10
    dense,_=clouds(430,512,256,.003);sparse,_=clouds(430,512,64,.003)
    assert np.array_equal(dense[:,:64],sparse)
    result=dict(seed_train=43,seed_test=430,seed_permutation=431,python=platform.python_version(),
        numpy=np.__version__,matplotlib=matplotlib.__version__,radii_m=radii,
        train_samples=1024,test_samples_per_condition=512,train_points=256,train_noise_m=.003,
        class_definition='two independent signs: local Gaussian bump and broad x slope',
        scores=scores,neighborhoods=group_stats,point_order_max_error=permutation_error,
        sparse_is_exact_dense_subset=True,parameter_counts={n:int(m[2].size) for n,m in models.items()},
        elapsed_seconds=time.perf_counter()-start,
        scope='Synthetic analytic local-plane features + linear ridge classifier; not FPN/PointNet++/NOCS/EPro-PnP training')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez(out/'arrays.npz',train_features=F,train_labels=y_train,test_labels=test_labels,**test_features)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    xx=np.arange(len(settings));width=.19
    for j,name in enumerate(specs):axes[0].bar(xx+(j-1.5)*width,[scores[c][name]['accuracy'] for c in settings],width,label=name)
    axes[0].set(xticks=xx,xticklabels=list(settings),ylim=(0,1.16),ylabel='Correct / 512',title='Paired tests; one frozen training set');axes[0].legend(ncols=2,fontsize=8,loc='upper center')
    axes[1].bar(list(settings),[group_stats[c]['small_insufficient_fraction'] for c in settings],color='#da6d3e')
    axes[1].set(ylim=(0,1),ylabel='Fraction with <3 small-neighborhood points',title='Missing local evidence under sparse sampling')
    fig.savefig(out/'multiscale_ablation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    p=train[0];sc=axes[0].scatter(p[:,0]*1000,p[:,1]*1000,c=p[:,2]*1000,cmap='coolwarm',s=12)
    for r,c in zip(radii,['#101820','#39a96b']):axes[0].add_patch(plt.Circle((0,0),r*1000,fill=False,color=c,lw=2))
    axes[0].axis('equal');axes[0].set(xlabel='x (mm)',ylabel='y (mm)',title='Same point cloud, two support radii');fig.colorbar(sc,ax=axes[0],label='z (mm)')
    for cls in range(4):
        mask=y_train==cls;axes[1].scatter(F[mask,0]*1000,F[mask,3],s=5,alpha=.4,label=f'class {cls}')
    axes[1].set(xlabel='Small-radius intercept (mm)',ylabel='Large-radius slope (m/m)',title='Local detail + broad context');axes[1].legend(markerscale=2)
    fig.savefig(out/'multiscale_features.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
