"""D37: grouped splitting, lazy loading and masked variable-length collation."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def split_frames(n, rng):
    p = rng.permutation(n)
    return [p[:int(.6*n)], p[int(.6*n):int(.8*n)], p[int(.8*n):]]


def nn_predict(x, y, train, test):
    distance = ((x[test,None,:] - x[None,train,:]) ** 2).sum(-1)
    return y[train[distance.argmin(1)]]


class TinyRGBD:
    """Each __getitem__ reads one mmap slice; no labels enter network inputs."""
    def __init__(self, root, ids):
        self.rgb = np.load(root/'rgb.npy', mmap_mode='r')
        self.depth = np.load(root/'depth.npy', mmap_mode='r')
        self.ids = np.asarray(ids)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        i = int(self.ids[index])
        rgb = self.rgb[i].astype(np.float32).transpose(2,0,1)/255
        depth = self.depth[i].astype(np.float32)*.001
        valid = depth > 0
        return dict(frame_id=i, rgb=rgb, depth=depth, valid=valid)


def pad_points(point_lists):
    b, n = len(point_lists), max(map(len, point_lists))
    points, valid = np.zeros((b,n,3),np.float32), np.zeros((b,n),bool)
    for i, p in enumerate(point_lists):
        points[i,:len(p)] = p
        valid[i,:len(p)] = True
    means = (points*valid[...,None]).sum(1)/valid.sum(1,keepdims=True)
    return points, valid, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jitter', type=float, default=.03)
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day37')
    a = ap.parse_args(); start = time.perf_counter(); out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(37)
    scene=np.repeat(np.arange(30),20); n=len(scene)
    background=rng.normal(size=(30,8))
    position=rng.uniform(-.15,.15,size=30)
    x=background[scene]+a.jitter*rng.normal(size=(n,8))
    y=position[scene]+rng.normal(0,.001,n)
    frame_split=split_frames(n,np.random.default_rng(370))
    groups=np.random.default_rng(371).permutation(30)
    group_split=[np.flatnonzero(np.isin(scene,g)) for g in [groups[:18],groups[18:24],groups[24:]]]
    results=[]
    for label, ids in [('frame_random',frame_split),('scene_grouped',group_split)]:
        train,val,test=ids
        pred=nn_predict(x,y,train,test)
        overlap=set(scene[train]) & set(scene[test])
        results.append(dict(split=label,train=len(train),val=len(val),test=len(test),
                            shared_train_test_scenes=len(overlap),
                            test_rmse_m=float(np.sqrt(np.mean((pred-y[test])**2)))))
    assert results[0]['shared_train_test_scenes'] > 0
    assert results[1]['shared_train_test_scenes'] == 0
    # Use a tiny disk fixture: 600 x 12 x 16 RGB-D, not a real dataset.
    rgb=np.zeros((n,12,16,3),np.uint8)
    rgb[:]=rng.integers(0,256,(n,1,1,3),dtype=np.uint8)
    depth=np.full((n,12,16),800,np.uint16); depth[:,0,:]=0
    np.save(out/'rgb.npy',rgb); np.save(out/'depth.npy',depth)
    train_ds=TinyRGBD(out,group_split[0]); samples=[train_ds[0],train_ds[1]]
    batch_rgb=np.stack([s['rgb'] for s in samples])
    batch_depth=np.stack([s['depth'] for s in samples])
    points,valid,mean=pad_points([np.array([[1,0,0],[3,0,0]]),np.array([[6,0,0]])])
    naive=points.mean(1)
    mu=x[group_split[0]].mean(0); std=x[group_split[0]].std(0)
    normalized=(x[group_split[2]]-mu)/np.maximum(std,1e-6)
    manifest={'train':group_split[0].tolist(),'val':group_split[1].tolist(),'test':group_split[2].tolist()}
    (out/'split.json').write_text(json.dumps(manifest,indent=2))
    result=dict(day=37,verified_on='2026-10-24',seed=37,split_seeds=[370,371],device='CPU',
                python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                data='30 synthetic scenes x 20 frames, random scene background and target',
                jitter=a.jitter,comparison=results,training='1-NN memorization, no parameter updates',
                batch_rgb_shape=list(batch_rgb.shape),batch_depth_shape=list(batch_depth.shape),
                invalid_depth_per_frame=int((~samples[0]['valid']).sum()),
                masked_centroid_x=mean[:,0].tolist(),naive_centroid_x=naive[:,0].tolist(),
                heldout_normalized_feature_mean=normalized.mean(0).tolist(),
                hardware_interactions=0,elapsed_s=time.perf_counter()-start)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    ax[0].bar(['frame random','scene grouped'],[r['test_rmse_m']*1000 for r in results],color=['#4c998b','#e49444'])
    ax[0].set_ylabel('Held-out position RMSE (mm)'); ax[0].set_title('Background shortcut with 1-NN')
    grid=np.zeros((30,20),int)
    for k,ids in enumerate(frame_split): grid.reshape(-1)[ids]=k
    ax[1].imshow(grid,aspect='auto',cmap='viridis',vmin=0,vmax=2)
    ax[1].set_xlabel('Frame within scene'); ax[1].set_ylabel('Scene'); ax[1].set_title('Random split mixes every video: 0/1/2')
    fig.savefig(out/'split_audit.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.6),constrained_layout=True)
    q=np.arange(2); ax.bar(q-.16,naive[:,0],.32,label='zero padding, unmasked')
    ax.bar(q+.16,mean[:,0],.32,label='valid-mask mean'); ax.set_xticks(q,['cloud A: 2 points','cloud B: 1 point'])
    ax.set_ylabel('Centroid x (synthetic units)'); ax.legend(); fig.savefig(out/'padding.png',dpi=160); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
