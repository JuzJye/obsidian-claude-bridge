"""D60 ridge alignment plus 1NN evaluation: groups and cross-modal mismatch."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_protocol(x,t,y,groups,split,shuffle):
    train,val,test=split
    mu,sd=t[train].mean(0),t[train].std(0)+1e-8
    a=np.column_stack([(t-mu)/sd,np.ones(len(t))]); lam=.01
    w=np.linalg.solve(a[train].T@a[train]+lam*np.eye(a.shape[1]),a[train].T@x[train])
    aligned=a@w; base=(x+aligned)/2
    def predict(ids,corrupt=False):
        pair=aligned[ids].copy()
        if corrupt: pair=pair[np.random.default_rng(660).permutation(len(ids))]
        f=(x[ids]+pair)/2
        dist=((f[:,None,:]-base[train][None,:,:])**2).sum(2)
        return y[train[dist.argmin(1)]],f
    pv,_=predict(val); pred,f=predict(test,shuffle)
    cm=np.zeros((2,2),int); np.add.at(cm,(y[test],pred),1)
    recall=np.diag(cm)/np.maximum(cm.sum(1),1)
    precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    per_group=np.array([(pred[groups[test]==g]==y[test][groups[test]==g]).mean() for g in np.unique(groups[test])])
    rng=np.random.default_rng(960); boot=rng.choice(per_group,(1000,len(per_group)),replace=True).mean(1)
    return {'train_n':len(train),'val_n':len(val),'test_n':len(test),'test_group_n':len(per_group),
      'train_test_group_overlap':len(set(groups[train])&set(groups[test])),
      'validation_accuracy':float((pv==y[val]).mean()),'correct':int((pred==y[test]).sum()),
      'accuracy':float((pred==y[test]).mean()),'balanced_accuracy':float(recall.mean()),'macro_f1':float(f1.mean()),
      'group_mean_accuracy':float(per_group.mean()),'group_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
      'alignment_rmse':float(np.sqrt(((aligned[test]-x[test])**2).mean())),
      'fusion_pair_rmse':float(np.sqrt(((2*f-x[test]-x[test])**2).mean())),
      'confusion':cm.tolist(),'per_group_accuracy':per_group.tolist()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shuffle-test-touch',action='store_true')
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day60/default')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); rng=np.random.default_rng(60)
    groups=np.repeat(np.arange(90),10); labels=rng.integers(0,2,90); y=labels[groups]
    identity=rng.normal(size=(90,5)); latent=np.column_stack([.35*(2*labels-1),identity])
    x=latent[groups]+rng.normal(0,.07,(900,6)); q,_=np.linalg.qr(rng.normal(size=(6,6))); t=latent[groups]@q+rng.normal(0,.07,(900,6))
    ids=rng.permutation(900); g=rng.permutation(90)
    splits={'frame_random':(ids[:600],ids[600:750],ids[750:]),
      'group_disjoint':tuple(np.flatnonzero(np.isin(groups,a)) for a in (g[:60],g[60:75],g[75:]))}
    results={k:run_protocol(x,t,y,groups,v,args.shuffle_test_touch) for k,v in splits.items()}
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,(name,r) in zip(axs,results.items()):
        ax.bar(['Val accuracy','Test accuracy','Macro F1'],[r['validation_accuracy'],r['accuracy'],r['macro_f1']]); ax.set(ylim=(0,1.08),title=name); ax.tick_params(axis='x',rotation=15)
    fig.tight_layout(); fig.savefig(args.out/'protocol.png',dpi=160); plt.close(fig)
    r=results['group_disjoint']; fig,axs=plt.subplots(1,2,figsize=(10,4)); cm=np.array(r['confusion']); axs[0].imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): axs[0].text(j,i,str(cm[i,j]),ha='center',va='center')
    axs[0].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='True',title='Independent groups, all 150 frames')
    axs[1].bar(range(len(r['per_group_accuracy'])),r['per_group_accuracy']); axs[1].set(xlabel='Held-out group index',ylabel='Accuracy',ylim=(0,1.05),title='Correlated frames stay in groups')
    fig.tight_layout(); fig.savefig(args.out/'independent.png',dpi=160); plt.close(fig)
    report={'seed':60,'python':platform.python_version(),'numpy':np.__version__,'shuffle_test_touch':args.shuffle_test_touch,
      'results':results,'runtime_s':time.perf_counter()-start,'data':'90 synthetic object groups; 10 correlated frames/group; no real benchmark',
      'robot_trials':0,'learning_status':'not_assessed'}
    (args.out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
