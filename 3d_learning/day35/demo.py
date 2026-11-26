"""D35: handwritten PnP, independent PointNet/BA audits, task interference.
Run from vault root: python -B 3d_learning/day35/demo.py
CPU only. Reuses teaching modules in day15/demo.py and day23/demo.py.
Does not execute their main functions or modify their saved results.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import platform
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_module(day):
    path = ROOT / f"3d_learning/day{day}/demo.py"
    spec = importlib.util.spec_from_file_location(f"course_day{day}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_so3(w):
    angle = np.linalg.norm(w)
    W = skew(w)
    if angle < 1e-8:
        return np.eye(3) + W + .5 * W @ W
    return np.eye(3) + np.sin(angle) / angle * W + (1 - np.cos(angle)) / angle**2 * W @ W


def project(X, R, t, K):
    q = X @ R.T + t
    if np.any(q[:, 2] <= 0):
        raise ValueError("Projection contains non-positive camera depth")
    uvw = q @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def dlt_pose(X, uv, K):
    """Calibrated, non-planar DLT initialization; no OpenCV PnP call."""
    if len(X) < 6 or np.linalg.matrix_rank(X - X.mean(0), tol=1e-8) < 3:
        raise ValueError("This teaching DLT requires >=6 non-coplanar points")
    center = X.mean(0)
    scale = np.sqrt(3) / np.mean(np.linalg.norm(X - center, axis=1))
    T = np.eye(4)
    T[:3, :3] *= scale
    T[:3, 3] = -scale * center
    Xh = np.c_[X, np.ones(len(X))] @ T.T
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    xy = rays[:, :2] / rays[:, 2:3]
    A = []
    for Xj, (x, y) in zip(Xh, xy):
        A.append(np.r_[Xj, np.zeros(4), -x * Xj])
        A.append(np.r_[np.zeros(4), Xj, -y * Xj])
    _, singular, vt = np.linalg.svd(np.array(A), full_matrices=False)
    P = vt[-1].reshape(3, 4) @ T
    if np.linalg.det(P[:, :3]) < 0:
        P *= -1
    U, values, Vt = np.linalg.svd(P[:, :3])
    R = U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
    t = P[:, 3] / values.mean()
    return R, t, singular


def pose_jacobian(X, R, t, K):
    rows = []
    for q in X @ R.T + t:
        x, y, z = q
        Jp = np.array([[K[0, 0] / z, 0, -K[0, 0] * x / z**2],
                       [0, K[1, 1] / z, -K[1, 1] * y / z**2]])
        rows.append(Jp @ np.c_[-skew(q), np.eye(3)])
    return np.vstack(rows)


def refine_pose(X, uv, K, R, t):
    """Damped Gauss-Newton with local left rotation and local translation."""
    damping, history = 1e-3, []
    for _ in range(35):
        r = (project(X, R, t, K) - uv).ravel()
        history.append(float(np.sqrt(np.mean(r**2))))
        J = pose_jacobian(X, R, t, K)
        H, g = J.T @ J, J.T @ r
        step = np.linalg.solve(H + damping * np.diag(np.maximum(np.diag(H), 1)), -g)
        Q = exp_so3(step[:3])
        nr, nt = Q @ R, Q @ t + step[3:]
        if np.all((X @ nr.T + nt)[:, 2] > .1):
            cost = np.sum((project(X, nr, nt, K) - uv)**2)
        else:
            cost = np.inf
        if cost < r @ r:
            R, t = nr, nt
            damping = max(1e-9, damping / 3)
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            damping = min(1e9, damping * 10)
    history.append(float(np.sqrt(np.mean((project(X, R, t, K) - uv)**2))))
    return R, t, np.array(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-px', type=float, default=.7)
    parser.add_argument('--out-dir', type=Path, default=ROOT / '_assets/3d_learning/day35')
    args = parser.parse_args()
    if args.noise_px < 0:
        raise ValueError('noise-px must be non-negative')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(35)
    rng = np.random.default_rng(35)
    K = np.array([[620., 0, 320.], [0, 610., 240.], [0, 0, 1.]])
    X = rng.uniform([-.3, -.2, -.15], [.3, .2, .15], (48, 3))
    Xt = rng.uniform([-.3, -.2, -.15], [.3, .2, .15], (120, 3))
    Rtrue, ttrue = exp_so3(np.array([.18, -.12, .07])), np.array([.03, -.02, 1.1])
    uv = project(X, Rtrue, ttrue, K) + args.noise_px * rng.normal(size=(48, 2))
    Rt, tt, singular = dlt_pose(X, uv, K)
    R, t, history = refine_pose(X, uv, K, Rt, tt)
    test_error = np.linalg.norm(project(Xt, R, t, K) - project(Xt, Rtrue, ttrue, K), axis=1)
    angle = np.degrees(np.arccos(np.clip((np.trace(R @ Rtrue.T) - 1) / 2, -1, 1)))
    J = pose_jacobian(X[:2], R, t, K)
    eps, numerical = 1e-6, []
    for k in range(6):
        step = np.zeros(6); step[k] = eps
        values = []
        for sign in [1, -1]:
            Q = exp_so3(sign * step[:3])
            values.append(project(X[:2], Q @ R, Q @ t + sign * step[3:], K).ravel())
        numerical.append((values[0] - values[1]) / (2 * eps))
    jac_error = float(np.max(np.abs(J - np.array(numerical).T)))
    planar = X.copy(); planar[:, 2] = 0
    try:
        dlt_pose(planar, uv, K)
        raise AssertionError('Planar data should be rejected by this implementation')
    except ValueError:
        planar_rejected = True
    # A metre/mm mistake leaves training projection invariant if ALL model points
    # and translation scale together; a mixed-unit downstream target reveals it.
    Rmm, tmm, _ = dlt_pose(X * 1000, uv, K)
    Rmm, tmm, _ = refine_pose(X * 1000, uv, K, Rmm, tmm)
    scale_projection_error = float(np.max(np.abs(project(X * 1000, Rmm, tmm, K) - project(X, R, t, K))))
    pnp = dict(train_points=48, heldout_points=120, noise_px=args.noise_px,
               initial_train_coordinate_rmse_px=float(history[0]), final_train_coordinate_rmse_px=float(history[-1]),
               heldout_euclidean_rms_px=float(np.sqrt(np.mean(test_error**2))),
               translation_error_mm=float(np.linalg.norm(t - ttrue) * 1000), rotation_error_deg=float(angle),
               analytic_jacobian_max_difference=jac_error, planar_input_rejected=planar_rejected,
               model_mm_translation_z=float(tmm[2]), model_m_translation_z=float(t[2]),
               simultaneous_scale_projection_max_difference_px=scale_projection_error)
    pn = load_module(15)
    points = torch.from_numpy(rng.normal(size=(3, 96, 3)).astype(np.float32))
    model = pn.TinyPointNet().eval()
    perm = torch.randperm(96)
    with torch.no_grad():
        logits, seg, _, _ = model(points)
        pl, ps, _, _ = model(points[:, perm])
    pointnet = dict(classification_permutation_max_difference=float((pl - logits).abs().max()),
                    segmentation_aligned_max_difference=float((ps - seg[:, perm]).abs().max()),
                    segmentation_unaligned_max_difference=float((ps - seg).abs().max()),
                    training_steps=0, input_shape=list(points.shape), source='day15/demo.py')
    ba = load_module(23)
    XX = rng.uniform([-.5, -.4, 3.2], [.5, .4, 4.8], (16, 3))
    RR = np.array([np.eye(3), np.eye(3), exp_so3(np.array([.025, -.03, .01]))])
    centers = np.array([[0., 0, 0], [.6, 0, 0], [.2, .35, .02]])
    tt0 = -np.einsum('nij,nj->ni', RR, centers)
    obs, hold = [], []
    for i in range(3):
        for j in range(16):
            pixel = ba.project(RR[i] @ XX[j] + tt0[i])
            if i == 2 and j % 4 == 0:
                hold.append((i, j, pixel))
            else:
                obs.append((i, j, pixel + rng.normal(0, .2, 2)))
    Xi = XX + rng.normal(0, .025, XX.shape)
    Ri, ti = RR.copy(), tt0.copy()
    Q = exp_so3(np.array([.012, -.009, .007]))
    Ri[2], ti[2] = Q @ Ri[2], Q @ ti[2] + np.array([.018, -.015, .02])
    _, JJ, rr = ba.linearize(Ri, ti, Xi, obs, False)
    schur, H, g = ba.schur_step(JJ, rr, 6, 1e-3)
    schur_diff = float(np.max(np.abs(schur - np.linalg.solve(H, -g))))
    rb, tb, xb, bh = ba.optimize(Ri, ti, Xi, obs, hold, False)
    ba_report = dict(source='day23/demo.py', fixed_metric_camera_poses=2, optimized_camera_poses=1,
                     points=16, train_observations=len(obs), heldout_observations=len(hold),
                     initial_heldout_coordinate_rmse_px=float(bh[0, 2]),
                     final_heldout_coordinate_rmse_px=float(bh[-1, 2]),
                     schur_direct_max_difference=schur_diff,
                     median_point_error_mm=float(np.median(np.linalg.norm(xb - XX, axis=1)) * 1000))
    # Deliberately conflicting tasks WITHOUT a task indicator: y=x vs y=-x.
    # Exact population E[x^2]=1; loss = E[(wx-y)^2]. This is not an EWC/MAML run.
    w, losses = 0., []
    for target in [1.] * 60 + [-1.] * 60:
        w -= .1 * 2 * (w - target)
        losses.append([(w - 1)**2, (w + 1)**2, w])
    losses = np.array(losses)
    continual = dict(task_A_loss_after_A=float(losses[59, 0]), task_A_loss_after_B=float(losses[-1, 0]),
                     task_B_loss_after_B=float(losses[-1, 1]), balanced_replay_optimal_w=0.,
                     balanced_replay_each_task_mse=1., task_conditioned_two_head_each_mse=0.,
                     warning='Opposite labels for the same input require task context or a distribution; replay is not magic.')
    alpha = 1 - np.exp(-np.array([2., 3.]) * .2)
    T = np.exp(-np.r_[0., np.cumsum(np.array([2., 3.]) * .2)])
    weights = T[:-1] * alpha
    conservation = float(abs(weights.sum() + T[-1] - 1))
    assert jac_error < 1e-6 and schur_diff < 1e-8 and conservation < 1e-12
    assert pointnet['classification_permutation_max_difference'] < 1e-6
    assert pointnet['segmentation_aligned_max_difference'] < 1e-6
    assert bh[-1, 2] < bh[0, 2] and history[-1] <= history[0] + 1e-8
    report = dict(date='2026-10-09', seed=35, device='cpu', python=platform.python_version(),
                  numpy=np.__version__, torch=torch.__version__, pnp=pnp, pointnet=pointnet,
                  ba=ba_report, continual=continual, rendering_mass_error=conservation,
                  seconds_main=float(time.perf_counter() - start),
                  scope='Synthetic module audits; no user assessment, trained PointNet accuracy, official VLA inference or robot execution.')
    (args.out_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(args.out_dir / 'arrays.npz', pnp_history=history, pnp_test_error=test_error,
             ba_history=bh, continual_losses=losses, pnp_uv=uv, pnp_pred=project(X, R, t, K))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
