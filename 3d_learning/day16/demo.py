"""D16 CPU mechanisms: FPS/grouping, frustum contamination and grasp contracts.
No trained PointNet++ / Frustum PointNets model, no robot control.
"""
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon


def fps(points, count):
    indices = []
    nearest_sq = np.full(len(points), np.inf)
    current = int(np.argmin(points[:, 0]))  # Geometric initial point, no ties here.
    for _ in range(count):
        indices.append(current)
        nearest_sq = np.minimum(nearest_sq, ((points - points[current]) ** 2).sum(1))
        current = int(np.argmax(nearest_sq))
    return np.array(indices)


def grouped_features(points, centroids, radius, cap=32):
    groups = np.zeros((len(centroids), cap, 3))
    valid = np.zeros((len(centroids), cap), dtype=bool)
    raw_counts = []
    for j, c in enumerate(centroids):
        d = np.linalg.norm(points - c, axis=1)
        ids = np.flatnonzero(d <= radius)
        ids = ids[np.argsort(d[ids])]
        raw_counts.append(len(ids))
        ids = ids[:cap]
        groups[j, :len(ids)] = points[ids] - c
        valid[j, :len(ids)] = True
    # Fixed educational pointwise feature; not learned weights.
    h = np.concatenate([groups, -groups, np.linalg.norm(groups, axis=-1, keepdims=True)], -1)
    h = np.maximum(h, 0)
    h[~valid] = -np.inf
    pooled = h.max(1)
    assert np.isfinite(pooled).all()  # Every center is itself an input point.
    return pooled, np.array(raw_counts), groups, valid


def cover(points, selected):
    return float(np.sqrt(((points[:, None] - selected[None]) ** 2).sum(-1).min(1).max()))


def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=16)
    p.add_argument('--out-dir', type=Path,
                   default=Path(__file__).resolve().parents[2] / '_assets/3d_learning/day16')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dense = np.column_stack([rng.uniform(-1, 0, 600), rng.uniform(-.5, .5, 600), np.zeros(600)])
    sparse = np.column_stack([rng.uniform(0, 1, 100), rng.uniform(-.5, .5, 100), np.zeros(100)])
    cloud = np.concatenate([dense, sparse])
    fids = fps(cloud, 24)
    rids = rng.choice(len(cloud), 24, replace=False)
    centers = cloud[fids]
    small, ns, groups, valid = grouped_features(cloud, centers, .15)
    large, nl, _, _ = grouped_features(cloud, centers, .35)
    msg = np.concatenate([small, large], -1)
    # Same cloud, centroids and cap; only neighborhood radius changes.
    assert (nl >= ns).all() and msg.shape == (24, 14)
    obj = np.column_stack([rng.uniform(-.10, .10, 240), rng.uniform(-.07, .07, 240), np.full(240, 1.10)])
    background = np.column_stack([rng.uniform(-.13, .13, 160), rng.uniform(-.09, .09, 160), np.full(160, 1.70)])
    pts = np.concatenate([obj, background])
    uv = pts[:, :2] / pts[:, 2, None] * 400 + [320, 240]
    roi = (uv[:, 0] >= 280) & (uv[:, 0] <= 360) & (uv[:, 1] >= 210) & (uv[:, 1] <= 270)
    truth_mask = np.arange(len(pts)) < len(obj)
    frustum = pts[roi]
    foreground = pts[roi & truth_mask]
    full_center = np.array([0., 0., 1.20])
    c_all, c_visible = frustum.mean(0), foreground.mean(0)
    target_residual = full_center - c_visible  # A supervision target, not a prediction.
    # Candidate coordinates use base z up, independently of optical camera above.
    T_bo = np.eye(4)
    T_bo[:3, :3] = Rz(np.deg2rad(30))
    T_bo[:3, 3] = [.50, .20, .12]
    specifications = [('wide-top', 0., 0., False), ('narrow-top', np.pi/2, 0., False),
                      ('narrow-near-obstacle', np.pi/2, .08, False),
                      ('narrow-from-below', np.pi/2, 0., True)]
    obstacle = np.array([.455, .305, .018])  # Base x,y and radius; tabletop top view.
    finger_radius = .008
    candidates = []
    for name, angle, offset, below in specifications:
        # +x of grasp is closing axis; +z is the direction of approach velocity.
        T_og = np.eye(4)
        xaxis = np.array([np.cos(angle), np.sin(angle), 0.])
        zaxis = np.array([0., 0., 1. if below else -1.])
        yaxis = np.cross(zaxis, xaxis)
        T_og[:3, :3] = np.column_stack([xaxis, yaxis, zaxis])
        T_og[:3, 3] = [0., offset, 0.]
        T_bg = T_bo @ T_og
        width = .10 * abs(np.cos(angle)) + .06 * abs(np.sin(angle)) + .008
        c = T_bg[:3, 3]
        axis = T_bg[:3, 0]
        fingers = np.stack([c[:2] - axis[:2]*width/2, c[:2] + axis[:2]*width/2])
        clearance = np.linalg.norm(fingers - obstacle[:2], axis=1).min() - obstacle[2] - finger_radius
        width_ok, approach_ok = width <= .09, not below
        center_ok = abs(offset) <= .01  # Template must straddle this centered object.
        valid_candidate = bool(width_ok and approach_ok and center_ok and clearance > 0)
        candidates.append({'name': name, 'T_base_grasp': T_bg.tolist(),
                           'width_m': float(width), 'clearance_m': float(clearance),
                           'center_ok': center_ok, 'width_ok': bool(width_ok),
                           'approach_ok': approach_ok, 'valid_in_toy_filter': valid_candidate,
                           'same_object_detection_score': .97})
    assert sum(c['valid_in_toy_filter'] for c in candidates) == 1
    report = {'seed': args.seed, 'python': platform.python_version(), 'numpy': np.__version__,
              'training_steps': 0, 'point_count': len(cloud),
              'fps_cover_radius': cover(cloud, cloud[fids]),
              'random_cover_radius': cover(cloud, cloud[rids]),
              'fps_sparse_centers': int((centers[:, 0] > 0).sum()),
              'random_sparse_centers': int((cloud[rids, 0] > 0).sum()),
              'neighborhood_radius_015_counts_min_median_max': [int(ns.min()), float(np.median(ns)), int(ns.max())],
              'neighborhood_radius_035_counts_min_median_max': [int(nl.min()), float(np.median(nl)), int(nl.max())],
              'group_shape': list(groups.shape), 'msg_shape': list(msg.shape),
              'frustum_point_count': len(frustum), 'oracle_foreground_count': len(foreground),
              'frustum_mean_center_error_m': float(np.linalg.norm(c_all-full_center)),
              'visible_mean_center_error_m': float(np.linalg.norm(c_visible-full_center)),
              'supervision_center_residual_m': target_residual.tolist(),
              'candidates': candidates,
              'scope': '2D finger clearance and simple template filters only; no IK, full hand/arm or contact simulation.'}
    (args.out_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, ids, title in zip(axes[:2], [rids, fids], ['Random: 24 centers', 'FPS: 24 centers']):
        ax.scatter(*cloud[:, :2].T, s=4, c='#a0aebb')
        ax.scatter(*cloud[ids, :2].T, s=35, c='#dc723a')
        ax.set(xlabel='x', ylabel='y', title=title, aspect='equal')
    axes[2].plot(ns, 'o-', label='radius 0.15')
    axes[2].plot(nl, 'o-', label='radius 0.35')
    axes[2].axhline(32, color='black', ls='--', label='cap=32')
    axes[2].set(xlabel='Same FPS center index', ylabel='Raw neighbor count', title='Radius controls geometric scale')
    axes[2].legend(fontsize=8)
    fig.savefig(args.out_dir/'neighborhoods.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].scatter(uv[len(obj):, 0], uv[len(obj):, 1], s=12, label='Background z=1.70 m')
    axes[0].scatter(uv[:len(obj), 0], uv[:len(obj), 1], s=9, label='Visible target z=1.10 m')
    axes[0].add_patch(Polygon([[280,210],[360,210],[360,270],[280,270]], fill=False, color='black'))
    axes[0].invert_yaxis()
    axes[0].set(xlabel='u (px)', ylabel='v (px)', title='One 2D ROI, multiple depths')
    axes[0].legend(fontsize=8)
    axes[1].scatter(pts[:,0], pts[:,2], c=truth_mask, cmap='coolwarm', s=10)
    axes[1].scatter([c_all[0], c_visible[0], full_center[0]], [c_all[2],c_visible[2],full_center[2]],
                    c=['black','#d89512','#178c6b'], marker='X', s=120)
    for c, label in [(c_all,'All ROI mean'),(c_visible,'Visible mean'),(full_center,'Full-box center')]:
        axes[1].annotate(label, (c[0],c[2]), xytext=(12,7), textcoords='offset points', fontsize=8)
    axes[1].set(xlabel='Camera x (m)', ylabel='Camera z (m)', title='Oracle mask still leaves visibility bias')
    axes[1].set_xlim(-.15,.24)
    fig.savefig(args.out_dir/'frustum.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
    local_rect = np.array([[-.05,-.03],[.05,-.03],[.05,.03],[-.05,.03]])
    rect = local_rect @ T_bo[:2,:2].T + T_bo[:2,3]
    for ax, c in zip(axes, candidates):
        T = np.array(c['T_base_grasp']); w = c['width_m']
        axis = T[:2,0]; center = T[:2,3]
        fingers = np.stack([center-axis*w/2, center+axis*w/2])
        ax.add_patch(Polygon(rect, color='#99bce0', alpha=.6))
        ax.add_patch(Circle(obstacle[:2], obstacle[2], color='#dd7560'))
        ax.plot(*fingers.T, 'o-', color='#20876d' if c['valid_in_toy_filter'] else '#be493a')
        ax.set(xlim=(.40,.64), ylim=(.11,.37), aspect='equal', xlabel='Base x (m)', ylabel='Base y (m)',
               title=c['name']+'\n'+('PASS toy filter' if c['valid_in_toy_filter'] else 'REJECT'))
        ax.text(.405,.115,f"width={w:.3f} m\nclearance={c['clearance_m']:.3f} m\ndetection score=0.97",fontsize=8)
    fig.savefig(args.out_dir/'grasp-candidates.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
