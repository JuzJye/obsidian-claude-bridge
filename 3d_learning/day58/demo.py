"""D58 synthetic tactile feature extraction; no force/hardware validation."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_data(seed, n=300, drift=0.):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    images = []
    for label in y:
        cx, cy = rng.uniform(-.25, .25, 2)
        patch = np.exp(-((xx-cx)**2+(yy-cy)**2)/.16)
        smooth = rng.uniform(.3,.6)*patch if label else np.zeros_like(xx)
        texture = .15*patch*np.sin(rng.uniform(22,32)*xx) if label == 2 else 0.
        images.append(.4 + smooth + texture + rng.normal(0,.008,xx.shape) + drift)
    return np.array(images), y


def features(images, border_correct):
    diff = images - .4
    if border_correct:
        border = np.concatenate([diff[:,0,:], diff[:,-1,:], diff[:,:,0], diff[:,:,-1]],1)
        diff = diff - np.median(border,1)[:,None,None]
    return np.column_stack([diff.mean((1,2)), diff.std((1,2)),
        np.abs(np.diff(diff,axis=1)).mean((1,2)),
        np.abs(np.diff(diff,axis=2)).mean((1,2))])


def train(x, y):
    mu, sd = x.mean(0), x.std(0)+1e-8
    z = np.column_stack([(x-mu)/sd, np.ones(len(x))])
    w = np.zeros((z.shape[1],3)); losses=[]
    target=np.eye(3)[y]
    for _ in range(700):
        logits=z@w; logits-=logits.max(1,keepdims=True)
        p=np.exp(logits); p/=p.sum(1,keepdims=True)
        losses.append(float(-np.log(p[np.arange(len(y)),y]+1e-12).mean()))
        w-=.08*(z.T@(p-target)/len(y)+.001*w)
    return w,mu,sd,losses


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--drift',type=float,default=0.)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day58/default')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    tr,yt=make_data(58); va,yv=make_data(158,150); te,ye=make_data(258,150,args.drift)
    results={}; models={}
    for name,correct in [('fixed_reference',False),('border_corrected',True)]:
        w,mu,sd,loss=train(features(tr,correct),yt)
        pred=lambda im: np.column_stack([(features(im,correct)-mu)/sd,np.ones(len(im))])@w
        pv=pred(va).argmax(1); pe=pred(te).argmax(1)
        cm=np.zeros((3,3),int); np.add.at(cm,(ye,pe),1)
        results[name]={'validation_correct':int((pv==yv).sum()),'validation_n':len(yv),
          'test_correct':int((pe==ye).sum()),'test_n':len(ye),'accuracy':float((pe==ye).mean()),
          'confusion':cm.tolist(),'loss_first':loss[0],'loss_last':loss[-1]}
        models[name]=(w,mu,sd,loss)
    fig,axs=plt.subplots(2,3,figsize=(10,6))
    for j in range(3):
        axs[0,j].imshow(te[j],cmap='viridis',vmin=.35,vmax=1.1); axs[0,j].set_title(['No contact','Smooth patch','Textured patch'][j]); axs[0,j].axis('off')
    axs[1,0].plot(models['fixed_reference'][3]); axs[1,0].set(xlabel='GD iteration',ylabel='Cross entropy',title='Actual softmax training')
    for j,name in enumerate(results):
        cm=np.array(results[name]['confusion']); axs[1,j+1].imshow(cm,cmap='Blues',vmin=0,vmax=50)
        for r in range(3):
            for c in range(3): axs[1,j+1].text(c,r,str(cm[r,c]),ha='center',va='center')
        axs[1,j+1].set(title=name,xlabel='Predicted',ylabel='True',xticks=range(3),yticks=range(3))
    fig.tight_layout(); fig.savefig(args.out/'features.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name,correct in [('fixed_reference',False),('border_corrected',True)]:
        w,mu,sd,_=models[name]; drift_values=np.linspace(0,.3,21); scores=[]
        for d in drift_values:
            im,y=make_data(258,150,d); z=np.column_stack([(features(im,correct)-mu)/sd,np.ones(len(y))]); scores.append(((z@w).argmax(1)==y).mean())
        ax.plot(drift_values,scores,label=name)
    ax.set(xlabel='Uniform test image brightness offset (unitless)',ylabel='Accuracy',ylim=(-.02,1.05),title='Same trained models and same test samples'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.out/'drift.png',dpi=160); plt.close(fig)
    report={'seed':[58,158,258],'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
      'drift':args.drift,'data':'independent synthetic patches; intensity is not force','n_train':300,'n_val':150,'n_test':150,
      'input_shape':[150,32,32],'feature_shape':[150,4],'results':results,'runtime_s':time.perf_counter()-start,
      'robot_trials':0,'learning_status':'not_assessed'}
    (args.out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
