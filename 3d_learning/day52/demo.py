"""D52: calibrated 6D PnP and pose visualization, synthetic known correspondences."""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def project(points, r, t, k):
    camera = points @ r.T + t
    assert np.all(camera[:, 2] > 0), 'points behind camera'
    homogeneous = camera @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def make_model(rng):
    # Unequal box dimensions and protrusion remove exact rotational symmetry.
    body = rng.uniform([-0.04, -0.06, -0.03], [0.04, 0.06, 0.03], (160, 3))
    faces = rng.integers(0, 3, len(body))
    signs = rng.choice([-1, 1], len(body))
    body[np.arange(len(body)), faces] = np.array([.04, .06, .03])[faces]*signs
    handle = rng.uniform([.04, -.025, -.01], [.08, .025, .01], (40, 3))
    return np.vstack((body, handle))


def errors(model, r, t, r_gt, t_gt, k):
    angle = np.degrees(np.arccos(np.clip((np.trace(r @ r_gt.T)-1)/2, -1, 1)))
    predicted = model @ r.T + t
    truth = model @ r_gt.T + t_gt
    return dict(rotation_deg=float(angle), translation_mm=float(np.linalg.norm(t-t_gt)*1000),
        add_mm=float(np.linalg.norm(predicted-truth, axis=1).mean()*1000),
        clean_reprojection_rmse_px=float(np.sqrt(np.mean(np.sum((project(model,r,t,k)-project(model,r_gt,t_gt,k))**2,axis=1)))))


def draw_axes(ax, r, t, k, style, alpha=1, length=.05):
    endpoints = np.vstack((np.zeros(3), np.eye(3)*length))
    pixels = project(endpoints, r, t, k)
    for i, color in enumerate(['red', 'green', 'blue']):
        ax.plot(pixels[[0, i+1], 0], pixels[[0, i+1], 1], style, color=color, lw=2, alpha=alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-px', type=float, default=.6)
    parser.add_argument('--seed', type=int, default=52)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day52')
    args = parser.parse_args()
    if args.noise_px < 0:
        raise ValueError('noise must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cv2.setRNGSeed(args.seed)
    model = make_model(rng)
    k = np.array([[570., 0, 320], [0, 570., 240], [0, 0, 1.]])
    r_gt = cv2.Rodrigues(np.array([.35, -.45, .20]))[0]
    t_gt = np.array([.035, -.015, .72])
    chosen = rng.choice(len(model), 90, replace=False)
    points = np.ascontiguousarray(model[chosen], dtype=np.float64)
    clean = project(points, r_gt, t_gt, k)
    observations = clean + rng.normal(size=clean.shape)*args.noise_px
    outlier_ids = rng.choice(len(points), 15, replace=False)
    observations[outlier_ids] += rng.uniform(-45, 45, (len(outlier_ids), 2))
    started = time.perf_counter()
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(points, observations, k, None,
        iterationsCount=500, reprojectionError=4., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError('PnP failed or insufficient inliers; do not output identity pose')
    ids = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(points[ids], observations[ids], k, None, rvec, tvec)
    r = cv2.Rodrigues(rvec)[0]; t = tvec.reshape(3)
    solver_ms = (time.perf_counter()-started)*1000
    assert np.allclose(r.T@r, np.eye(3), atol=1e-8) and np.linalg.det(r) > .999
    residuals = np.linalg.norm(project(points, r, t, k)-observations, axis=1)
    heldout = np.setdiff1d(np.arange(len(model)), chosen)
    heldout_rmse = np.sqrt(np.mean(np.sum((project(model[heldout],r,t,k)-project(model[heldout],r_gt,t_gt,k))**2,axis=1)))
    cases = {'estimated': (r, t), 'transpose_bug': (r.T, t),
             'z_plus_20mm': (r, t+np.array([0, 0, .02]))}
    comparison = {name: errors(model, rr, tt, r_gt, t_gt, k) for name, (rr,tt) in cases.items()}
    result = dict(seed=args.seed, noise_px=args.noise_px, correspondences=len(points),
        injected_outliers=len(outlier_ids), ransac_inliers=len(ids), ransac_threshold_px=4,
        inlier_rmse_px=float(np.sqrt(np.mean(residuals[ids]**2))),
        all_observation_rmse_px=float(np.sqrt(np.mean(residuals**2))),
        heldout_clean_rmse_px=float(heldout_rmse), cases=comparison, solver_ms=solver_ms,
        input_contract='known synthetic model and 2D-3D identities; no correspondence network',
        frame='T_CO: object -> camera optical (x right, y down, z forward)',
        length_unit='m', timestamp_s=0., physical_robot_trials=0, training_samples=0,
        runtime=dict(python=platform.python_version(), numpy=np.__version__,opencv=cv2.__version__,matplotlib=matplotlib.__version__))
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    transform = np.eye(4); transform[:3, :3]=r; transform[:3, 3]=t
    np.savez_compressed(args.out/'results.npz', model_points=model, K=k, R=r, t=t,
        R_gt=r_gt, t_gt=t_gt, T_CO=transform, correspondences_3d=points,
        observations_2d=observations, inlier_indices=ids, timestamp_s=np.array(0.))
    fig=plt.figure(figsize=(11,4.5)); ax=fig.add_subplot(121)
    gt_pixels=project(model,r_gt,t_gt,k); pred_pixels=project(model,r,t,k)
    ax.scatter(gt_pixels[:,0],gt_pixels[:,1],s=8,c='lightgray',label='GT model')
    ax.scatter(observations[:,0],observations[:,1],s=12,c='orange',label='observed landmarks')
    ax.scatter(pred_pixels[:,0],pred_pixels[:,1],s=3,c='black',label='estimated model')
    corners=np.array([[x,y,z] for x in [-.04,.08] for y in [-.06,.06] for z in [-.03,.03]])
    projected_corners=project(corners,r,t,k)
    for i in range(8):
        for j in range(i+1,8):
            if np.count_nonzero(corners[i]!=corners[j])==1:
                ax.plot(projected_corners[[i,j],0],projected_corners[[i,j],1],color='teal',lw=.8)
    draw_axes(ax,r_gt,t_gt,k,'--',.5); draw_axes(ax,r,t,k,'-')
    ax.set(xlabel='u [px]',ylabel='v [px]',title='PnP overlay: GT dashed / estimate solid',xlim=(230,430),ylim=(330,140))
    ax.set_aspect('equal'); ax.legend(fontsize=7)
    ax=fig.add_subplot(122,projection='3d'); camera=model@r.T+t
    ax.scatter(camera[:,0],camera[:,1],camera[:,2],s=5,color='teal',alpha=.5)
    for i,color in enumerate(['red','green','blue']):
        direction=r[:,i]*.05
        ax.quiver(*t,*direction,color=color,arrow_length_ratio=.15)
    ax.set(xlabel='camera X [m]',ylabel='camera Y [m]',zlabel='camera Z [m]',title='Object axes expressed in camera frame')
    ax.set_box_aspect((1,1,1)); fig.tight_layout(); fig.savefig(args.out/'pose_overlay.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].hist(residuals,bins=20,color='slateblue')
    axes[0].axvline(4,color='red',ls='--',label='RANSAC selection threshold')
    axes[0].set(xlabel='observed residual [px]',ylabel='count',title='Outliers remain in raw observations'); axes[0].legend(fontsize=6)
    for name,(rr,tt) in cases.items():
        err=np.linalg.norm(project(model,rr,tt,k)-project(model,r_gt,t_gt,k),axis=1)
        axes[1].plot(np.sort(err),np.arange(1,len(err)+1)/len(err),label=name)
    axes[1].set(xlabel='clean model reprojection error [px]',ylabel='CDF',title='Held geometry reveals visualization errors'); axes[1].legend(fontsize=7)
    axes[2].scatter(gt_pixels[:,0],gt_pixels[:,1],s=6,c='lightgray')
    draw_axes(axes[2],r,t,k,'-'); draw_axes(axes[2],r.T,t,k,':')
    axes[2].set(xlabel='u [px]',ylabel='v [px]',title='R solid vs incorrect R.T dotted',xlim=(230,430),ylim=(330,140))
    axes[2].set_aspect('equal'); fig.tight_layout(); fig.savefig(args.out/'pose_diagnostics.png',dpi=150); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
