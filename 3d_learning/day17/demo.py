"""D17: transparent toy 3D detection evaluation and 2D RRT/RRT-Connect.
Axis-aligned boxes, one frame/one class, all-points interpolated AP: NOT KITTI AP_R40.
Planning is CPU-only and has analytic segment/AABB collision checks.
"""
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def iou3d(a, b):
    overlap = np.maximum(0., np.minimum(a[:3]+a[3:]/2, b[:3]+b[3:]/2)
                         - np.maximum(a[:3]-a[3:]/2, b[:3]-b[3:]/2))
    intersection = np.prod(overlap)
    return float(intersection/(np.prod(a[3:])+np.prod(b[3:])-intersection))


def evaluate(boxes, scores, gt, threshold):
    order = np.argsort(-scores, kind='stable')
    used = np.zeros(len(gt), dtype=bool)
    tp, matches = [], []
    for i in order:
        overlaps = np.array([iou3d(boxes[i], g) for g in gt])
        overlaps[used] = -1
        j = int(overlaps.argmax())
        good = overlaps[j] >= threshold
        tp.append(int(good)); matches.append(j if good else -1)
        if good:
            used[j] = True
    tp = np.array(tp)
    recall = np.cumsum(tp)/len(gt)
    precision = np.cumsum(tp)/np.arange(1,len(tp)+1)
    r = np.r_[0., recall, 1.]
    p = np.r_[0., precision, 0.]
    for i in range(len(p)-2,-1,-1):
        p[i] = max(p[i],p[i+1])
    changes = np.flatnonzero(r[1:] != r[:-1])
    ap = float(np.sum((r[changes+1]-r[changes])*p[changes+1]))
    return {'threshold': threshold, 'order': order.tolist(), 'tp': tp.tolist(),
            'matches': matches, 'precision': precision.tolist(), 'recall': recall.tolist(),
            'ap_all_points': ap, 'final_recall': float(recall[-1]), 'final_precision': float(precision[-1])}


def nms(boxes, scores, threshold):
    remaining = list(np.argsort(-scores, kind='stable'))
    kept = []
    while remaining:
        i = remaining.pop(0); kept.append(i)
        remaining = [j for j in remaining if iou3d(boxes[i],boxes[j]) <= threshold]
    return np.array(kept)


def hits_rect(a, b, rect):
    # Exact segment/AABB slab test, including endpoints and tangency as collision.
    low, high = 0., 1.
    for k in range(2):
        d = b[k]-a[k]
        if abs(d) < 1e-12:
            if a[k] < rect[k] or a[k] > rect[k+2]:
                return False
        else:
            t1, t2 = (rect[k]-a[k])/d, (rect[k+2]-a[k])/d
            low, high = max(low,min(t1,t2)), min(high,max(t1,t2))
            if low > high:
                return False
    return True


class World:
    def __init__(self, robot_radius=.03, blocked=False):
        self.radius = robot_radius
        self.physical = np.array([[.43, 0., .57, 1. if blocked else .65]])
        self.inflated = self.physical + np.array([-robot_radius,-robot_radius,robot_radius,robot_radius])
        self.checks = 0

    def free(self, a, b):
        self.checks += 1
        if np.any(np.r_[a,b] < self.radius) or np.any(np.r_[a,b] > 1-self.radius):
            return False
        return not any(hits_rect(a,b,r) for r in self.inflated)


def trace(nodes, parents, index):
    out = []
    while index >= 0:
        out.append(nodes[index]); index = parents[index]
    return out[::-1]


def extend(nodes, parents, target, world, step):
    arr = np.array(nodes)
    i = int(np.linalg.norm(arr-target,axis=1).argmin())
    delta = target-arr[i]; distance = np.linalg.norm(delta)
    if distance < 1e-10:
        return 'reached', i
    q = arr[i]+delta*min(1.,step/distance)
    if not world.free(arr[i],q):
        return 'trapped', i
    nodes.append(q); parents.append(i)
    return ('reached' if distance <= step else 'advanced'), len(nodes)-1


def plan(start, goal, world, seed, bidirectional, budget=2500, step=.08):
    rng = np.random.default_rng(seed)
    if not world.free(start,start) or not world.free(goal,goal):
        return None, [], 0
    if bidirectional:
        trees = [([start],[-1]),([goal],[-1])]
        active = 0
        for iteration in range(1,budget+1):
            a,b = trees[active],trees[1-active]
            sample = rng.uniform(world.radius,1-world.radius,2)
            status, ia = extend(*a,sample,world,step)
            if status != 'trapped':
                while True:
                    status_b, ib = extend(*b,a[0][ia],world,step)
                    if status_b != 'advanced':
                        break
                if status_b == 'reached':
                    left,right = trace(*a,ia),trace(*b,ib)
                    path = left + right[-2::-1] if active == 0 else right + left[-2::-1]
                    return np.array(path), trees, iteration
            active = 1-active
        return None,trees,budget
    nodes,parents = [start],[-1]
    for iteration in range(1,budget+1):
        sample = goal if rng.random() < .10 else rng.uniform(world.radius,1-world.radius,2)
        status, index = extend(nodes,parents,sample,world,step)
        if status != 'trapped' and np.linalg.norm(nodes[index]-goal) <= step and world.free(nodes[index],goal):
            nodes.append(goal);parents.append(index)
            return np.array(trace(nodes,parents,len(nodes)-1)), [(nodes,parents)], iteration
    return None,[(nodes,parents)],budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--out-dir',type=Path,
                        default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day17')
    args = parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    gt = np.array([[0,0,0,1,1,1],[2,0,0,1,1,1],[4,0,0,1,1,1]],dtype=float)
    boxes = np.array([[0,0,0,1,1,1],[.02,0,0,1,1,1],[8,0,0,1,1,1],
                      [2,0,0,1,1,1],[4.3,0,0,1,1,1]],dtype=float)
    scores = np.array([.95,.93,.90,.85,.70])
    base = evaluate(boxes,scores,gt,.5)
    strict = evaluate(boxes,scores,gt,.7)
    rank_scores = scores.copy(); rank_scores[2]=.99
    bad_rank = evaluate(boxes,rank_scores,gt,.5)
    keep = nms(boxes,scores,.8)
    after_nms = evaluate(boxes[keep],scores[keep],gt,.5)
    assert np.isclose(base['ap_all_points'],11/15)
    assert base['tp'] == [1,0,0,1,1] and strict['tp'] == [1,0,0,1,0]
    start,goal = np.array([.10,.20]),np.array([.90,.20])
    planning, plots = {}, []
    for name,bidir,blocked,radius in [('rrt',False,False,.03),('rrt_connect',True,False,.03),
                                      ('blocked',True,True,.03),('radius_020',True,False,.20)]:
        world = World(radius,blocked)
        # Radius ablation needs valid endpoints for both robot sizes.
        s,g = (np.array([.22,.22]),np.array([.78,.22])) if name=='radius_020' else (start,goal)
        path,trees,iterations = plan(s,g,world,args.seed,bidir,budget=2500)
        if path is not None:
            assert np.allclose(path[0],s) and np.allclose(path[-1],g)
            assert all(world.free(a,b) for a,b in zip(path[:-1],path[1:]))
        planning[name] = {'found': path is not None, 'iterations': iterations,
                          'edge_checks':world.checks,'nodes':sum(len(t[0]) for t in trees),
                          'path_length': None if path is None else float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                          'robot_radius':radius}
        plots.append((name,world,s,g,path,trees))
    # Controlled radius comparison uses exactly the same start and goal as radius_020.
    smallworld=World(.03,False)
    smallpath,_,smalliters=plan(np.array([.22,.22]),np.array([.78,.22]),smallworld,args.seed,True)
    assert smallpath is not None
    planning['radius_003_same_endpoints']={'found':True,'iterations':smalliters,
        'path_length':float(np.linalg.norm(np.diff(smallpath,axis=0),axis=1).sum()),'robot_radius':.03}
    assert planning['rrt']['found'] and planning['rrt_connect']['found']
    assert not planning['blocked']['found'] and not planning['radius_020']['found']
    # Endpoint-only test misses an obstacle between two safe endpoints.
    testworld=World(.03)
    assert testworld.free(start,start) and testworld.free(goal,goal) and not testworld.free(start,goal)
    report={'seed':args.seed,'python':platform.python_version(),'numpy':np.__version__,
            'protocol':'one frame / one class / axis-aligned 3D IoU / greedy one-to-one / all-points AP',
            'base':base,'iou_threshold_070':strict,'only_fp_score_changed':bad_rank,
            'after_nms':after_nms,'nms_kept_prediction_indices':keep.tolist(),
            'offset_030_iou':iou3d(boxes[-1],gt[-1]),'planning':planning,
            'scope':'No training; no official dataset evaluation; 2D holonomic disc with conservative AABB inflation.'}
    (args.out_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for data,label in [(base,'IoU >= 0.5'),(strict,'IoU >= 0.7'),(bad_rank,'FP ranked first'),(after_nms,'After toy NMS')]:
        axes[0].step(data['recall'],data['precision'],where='post',marker='o',label=f"{label}; AP={data['ap_all_points']:.3f}")
    axes[0].set(xlabel='Recall',ylabel='Raw precision',xlim=(0,1.03),ylim=(0,1.05),title='AP uses the precision envelope')
    axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(5),base['tp'],label='TP at IoU=0.5',color='#32988b')
    for i,s in enumerate(scores):axes[1].text(i,.12,f's={s:.2f}',ha='center')
    axes[1].set_xticks(np.arange(5),['GT0','Duplicate','Background','GT1','GT2 shifted'])
    axes[1].set(ylabel='TP indicator',title='One ground truth can be matched only once',ylim=(0,1.2))
    fig.savefig(args.out_dir/'evaluation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for ax,(name,world,s,g,path,trees) in zip(axes.flat,plots):
        for rect in world.inflated:ax.add_patch(Rectangle(rect[:2],*(rect[2:]-rect[:2]),color='#f0b0a1',alpha=.55))
        for rect in world.physical:ax.add_patch(Rectangle(rect[:2],*(rect[2:]-rect[:2]),color='#a4493c',alpha=.7))
        for ti,(nodes,parents) in enumerate(trees):
            for j,parent in enumerate(parents):
                if parent>=0:
                    edge=np.array([nodes[parent],nodes[j]])
                    ax.plot(*edge.T,lw=.4,color=['#5c9ba9','#c69c4d'][ti%2],alpha=.3)
        if path is not None:ax.plot(*path.T,color='#1b755f',lw=2.5)
        ax.plot([s[0],g[0]],[s[1],g[1]],'k--',lw=.7)
        ax.scatter(*s,c='#207c55',s=60);ax.scatter(*g,c='#913d91',s=60)
        ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='x (m)',ylabel='y (m)',
               title=f"{name}: {'found' if path is not None else 'not found'}; r={world.radius}")
    fig.savefig(args.out_dir/'planning.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
