"""D63: train paired encoders; frozen probe and unseen-pair retrieval. CPU."""
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shuffle-pairs',action='store_true')
    ap.add_argument('--out',type=Path,default=Path('_assets/3d_learning/day63/default'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    torch.set_num_threads(2); torch.manual_seed(63); rng=np.random.default_rng(63)
    # Each row is an independent synthetic episode, not adjacent frames from one run.
    z=rng.normal(size=(768,4)).astype('float32')
    av=rng.normal(size=(4,6)); at=rng.normal(size=(4,6))
    v=z@av+.04*rng.normal(size=(768,6)); t=z@at+.04*rng.normal(size=(768,6))
    v=(v-v[:512].mean(0))/v[:512].std(0); t=(t-t[:512].mean(0))/t[:512].std(0)
    v=torch.tensor(v,dtype=torch.float32); t=torch.tensor(t,dtype=torch.float32)
    labels=torch.tensor((z[:,0]>.0).astype('int64'))
    ev=nn.Sequential(nn.Linear(6,24),nn.Tanh(),nn.Linear(24,8))
    et=nn.Sequential(nn.Linear(6,24),nn.Tanh(),nn.Linear(24,8))
    opt=torch.optim.Adam(list(ev.parameters())+list(et.parameters()),lr=.003)
    # A dedicated generator keeps probe initialization identical in both runs.
    pairing=(torch.randperm(512, generator=torch.Generator().manual_seed(630))
             if args.shuffle_pairs else torch.arange(512))
    train_curve=[]; val_curve=[]
    def embed(x,encoder): return F.normalize(encoder(x),dim=1)
    def contrast(a,b):
        score=a@b.T/.1; ids=torch.arange(len(a))
        return (F.cross_entropy(score,ids)+F.cross_entropy(score.T,ids))/2
    for epoch in range(250):
        opt.zero_grad(); loss=contrast(embed(v[:512],ev),embed(t[pairing],et))
        loss.backward(); opt.step(); train_curve.append(float(loss.detach()))
        with torch.no_grad():
            val_curve.append(float(contrast(embed(v[512:640],ev),embed(t[512:640],et))))
    with torch.no_grad(): zv=embed(v,ev); zt=embed(t,et)
    # Frozen visual encoder; contact labels are a separate downstream probe signal.
    probe=nn.Linear(8,2); po=torch.optim.Adam(probe.parameters(),lr=.03)
    for _ in range(250):
        po.zero_grad(); loss=F.cross_entropy(probe(zv[:512]),labels[:512]); loss.backward(); po.step()
    with torch.no_grad():
        sim=zv[640:]@zt[640:].T; hit=(sim.argmax(1)==torch.arange(128))
        pred=probe(zv[640:]).argmax(1); truth=labels[640:]
        confusion=torch.bincount(2*truth+pred,minlength=4).reshape(2,2)
        std=zv[640:].std(0); acc=float((pred==truth).float().mean())
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':63,'python':platform.python_version(),
            'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','shuffle_pairs':args.shuffle_pairs,
            'train_val_test_episodes':[512,128,128],'encoder_epochs':250,'probe_epochs':250,'temperature':.1,
            'encoder_parameters':sum(p.numel() for m in (ev,et) for p in m.parameters()),
            'final_training_loss':train_curve[-1],'final_validation_true_pair_loss':val_curve[-1],
            'test_retrieval_correct':int(hit.sum()),'test_retrieval_denominator':128,
            'test_retrieval_R1':float(hit.float().mean()),'test_probe_correct':int((pred==truth).sum()),
            'test_probe_denominator':128,'test_probe_accuracy':acc,'test_confusion':confusion.tolist(),
            'test_embedding_std_each_dim':std.tolist(),'robot_attempts':0,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':'Synthetic six-dimensional sensor vectors; not tactile image learning or official data validation.'}
    arrays={'visual':v.numpy(),'tactile':t.numpy(),'labels':labels.numpy(),'test_similarity':sim.numpy(),
            'train_loss':np.array(train_curve),'val_loss':np.array(val_curve),'train_pairing':pairing.numpy()}
    for name,module in [('visual_encoder',ev),('tactile_encoder',et),('probe',probe)]:
        arrays.update({name+'_'+key:val.detach().numpy() for key,val in module.state_dict().items()})
    np.savez_compressed(args.out/'results.npz',**arrays)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.plot(train_curve,label='Training pairs'); ax.plot(val_curve,label='True held-out pairs')
    ax.set(xlabel='Epoch',ylabel='Symmetric contrastive loss',title='Shuffled training pairs' if args.shuffle_pairs else 'Correct training pairs')
    ax.legend(); fig.savefig(args.out/'learning.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    im=ax[0].imshow(sim.numpy()[:32,:32],vmin=-1,vmax=1,cmap='coolwarm'); fig.colorbar(im,ax=ax[0])
    ax[0].set(xlabel='Tactile candidate ID',ylabel='Visual query ID',title='First 32 unseen pairs')
    ax[1].imshow(confusion.numpy(),cmap='Blues')
    for i in range(2):
        for j in range(2): ax[1].text(j,i,str(int(confusion[i,j])),ha='center',va='center')
    ax[1].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted contact',ylabel='True contact',title='Frozen visual encoder probe')
    fig.savefig(args.out/'retrieval_probe.png',dpi=160); plt.close(fig)
    report['wall_time_s']=time.perf_counter()-start
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
