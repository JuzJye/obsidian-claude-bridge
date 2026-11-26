"""D59 CPU cross-modal InfoNCE training on synthetic sensor features."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shuffle-pairs',action='store_true')
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day59/default')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    torch.set_num_threads(1); torch.manual_seed(59); rng=np.random.default_rng(59)
    a=rng.normal(size=(4,12)); b=rng.normal(size=(4,10))
    def sample(n):
        z=rng.normal(size=(n,4)); return [torch.tensor(v,dtype=torch.float32) for v in (z@a+.04*rng.normal(size=(n,12)),z@b+.04*rng.normal(size=(n,10)))]
    x,y=sample(256); vx,vy=sample(96); tx,ty=sample(96)
    mx,sx=x.mean(0),x.std(0); my,sy=y.mean(0),y.std(0)
    x,vx,tx=[(v-mx)/sx for v in (x,vx,tx)]; y,vy,ty=[(v-my)/sy for v in (y,vy,ty)]
    # Independent permutation RNG keeps encoder initialization identical in the ablation.
    if args.shuffle_pairs: y=y[torch.randperm(len(y), generator=torch.Generator().manual_seed(159))]
    enc_v=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,8))
    enc_t=nn.Sequential(nn.Linear(10,32),nn.Tanh(),nn.Linear(32,8))
    opt=torch.optim.Adam(list(enc_v.parameters())+list(enc_t.parameters()),lr=.006)
    def score(v,t): return F.normalize(enc_v(v),dim=1)@F.normalize(enc_t(t),dim=1).T
    def recall(s,k=1): return (s.topk(k,dim=1).indices==torch.arange(len(s))[:,None]).any(1).float().mean().item()
    losses=[]; vals=[]
    for step in range(350):
        logits=score(x,y)/.12; labels=torch.arange(len(x))
        loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        with torch.no_grad(): vals.append(recall(score(vx,vy)))
    with torch.no_grad():
        s=score(tx,ty); r1=recall(s); r5=recall(s,5); matrix=s.numpy()
    fig,axs=plt.subplots(1,2,figsize=(10,4)); axs[0].plot(losses); axs[0].set(xlabel='Step',ylabel='Symmetric InfoNCE',title='Training objective')
    axs[1].plot(vals); axs[1].axhline(1/96,color='gray',ls='--',label='Random R@1'); axs[1].set(xlabel='Step',ylabel='Recall@1',ylim=(0,1.05),title='Disjoint validation pairs'); axs[1].legend()
    fig.tight_layout(); fig.savefig(args.out/'training.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4)); im=axs[0].imshow(matrix[:24,:24],cmap='coolwarm',vmin=-1,vmax=1); axs[0].set(xlabel='Touch candidate',ylabel='Vision query',title='Test pair cosine similarity'); fig.colorbar(im,ax=axs[0])
    off=matrix[~np.eye(len(matrix),dtype=bool)]; axs[1].hist(off,bins=35,alpha=.5,density=True,label='Nonpaired'); axs[1].hist(np.diag(matrix),bins=15,alpha=.5,density=True,label='Paired'); axs[1].legend(); axs[1].set(xlabel='Cosine similarity',ylabel='Density',title='Held-out feature pairs')
    fig.tight_layout(); fig.savefig(args.out/'retrieval.png',dpi=160); plt.close(fig)
    report={'seed':59,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','shuffle_pairs':args.shuffle_pairs,
      'n_train':256,'n_val':96,'n_test':96,'vision_dim':12,'touch_dim':10,'embedding_dim':8,'temperature':.12,'steps':350,
      'loss_first':losses[0],'loss_last':losses[-1],'validation_r1':vals[-1],'test_r1':r1,'test_r5':r5,
      'paired_cosine':float(np.diag(matrix).mean()),'nonpaired_cosine':float(off.mean()),'runtime_s':time.perf_counter()-start,
      'data':'shared latent synthetic feature vectors; no images or robot actions','robot_trials':0,'learning_status':'not_assessed'}
    (args.out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
