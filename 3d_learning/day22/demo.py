"""D22: synthetic sparse SfM interfaces; no real image detector/COLMAP run.
Column-vector convention q=R_cw X_w+t_cw; metres and pixels. Seed 22.
"""
from pathlib import Path
import json
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day22"
K = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1]])


def project(X, R, t):
    q = X @ R.T + t
    h = q @ K.T
    return h[:, :2] / h[:, 2:3]


def triangulate(uv0, uv1, P0, P1):
    """Handwritten homogeneous DLT: four rows per two-view track."""
    points = []
    for (u, v), (a, b) in zip(uv0, uv1):
        A = np.stack([u * P0[2] - P0[0], v * P0[2] - P0[1],
                      a * P1[2] - P1[0], b * P1[2] - P1[1]])
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]
        points.append(h[:3] / h[3])
    return np.array(points)


def match_descriptors(a, b, ratio=0.8):
    """Euclidean ratio test + mutual nearest neighbour, not a detector."""
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    order = np.argsort(dist, axis=1)
    pairs = [(i, j) for i, (j, k) in enumerate(order[:, :2])
             if dist[i, j] < ratio * dist[i, k] and dist[:, j].argmin() == i]
    return np.asarray(pairs, dtype=int)


def rot_error(R, truth):
    return float(np.degrees(np.arccos(np.clip((np.trace(R @ truth.T)-1)/2, -1, 1))))


def refit_essential(a, b):
    """Calibrated eight-point fit on the consensus, enforce singular values (s,s,0)."""
    ah = np.column_stack([a, np.ones(len(a))]) @ np.linalg.inv(K).T
    bh = np.column_stack([b, np.ones(len(b))]) @ np.linalg.inv(K).T
    design = np.einsum("ni,nj->nij", bh, ah).reshape(-1, 9)
    _, _, vt = np.linalg.svd(design, full_matrices=False)
    u, s, vt = np.linalg.svd(vt[-1].reshape(3, 3))
    return u @ np.diag([np.mean(s[:2]), np.mean(s[:2]), 0.]) @ vt


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(22)
    cv2.setRNGSeed(22)
    n = 180
    X = rng.uniform([-1., -.7, 3.], [1., .7, 6.], (n, 3))
    R = np.stack([np.eye(3), cv2.Rodrigues(np.array([0., -.04, .01]))[0],
                  cv2.Rodrigues(np.array([.02, -.07, -.01]))[0]])
    C = np.array([[0., 0, 0], [.45, .02, 0], [.85, -.08, .03]])
    t = -np.einsum("nij,nj->ni", R, C)
    uv = np.array([project(X, r, tt) for r, tt in zip(R, t)])
    noisy = uv + rng.normal(0, .30, uv.shape)
    desc = rng.normal(size=(n, 32))
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    descriptors = np.array([desc + rng.normal(0, .01, desc.shape) for _ in range(3)])
    # Confident but wrong appearance: first 54 identities are cyclically permuted.
    descriptors[1, :54] = np.roll(descriptors[1, :54], 1, axis=0)
    pairs = match_descriptors(descriptors[0], descriptors[1])
    a, b = noisy[0, pairs[:, 0]], noisy[1, pairs[:, 1]]
    E, mask = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=.999,
                                 threshold=0.6)
    if E is None or E.shape != (3, 3):
        raise RuntimeError("No unique essential matrix returned")
    E = refit_essential(a[mask.ravel()!=0], b[mask.ravel()!=0])
    _, Rest, tunit, front = cv2.recoverPose(E, a, b, K, mask=mask.copy())
    keep = front.ravel() != 0
    accepted = pairs[keep]
    # Translation norm is unavailable to monocular images. This known baseline is
    # an EXTERNAL metric anchor in this experiment, also used to report metre errors.
    baseline = float(np.linalg.norm(C[1] - C[0]))
    test = tunit.ravel() * baseline
    P0 = K @ np.column_stack([np.eye(3), np.zeros(3)])
    P1 = K @ np.column_stack([Rest, test])
    Xest = triangulate(a[keep], b[keep], P0, P1)
    pair02 = dict(match_descriptors(descriptors[0], descriptors[2]).tolist())
    ids = np.array([k for k, (i, _) in enumerate(accepted) if i in pair02])
    uv2 = np.array([noisy[2, pair02[accepted[k, 0]]] for k in ids])
    ok, rv2, tv2, pnp_in = cv2.solvePnPRansac(Xest[ids], uv2, K, None,
        iterationsCount=1000, reprojectionError=2., confidence=.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("Third-view PnP failed")
    R2est = cv2.Rodrigues(rv2)[0]
    C2est = -R2est.T @ tv2.ravel()
    known_good = accepted[:, 0] == accepted[:, 1]
    errors = np.linalg.norm(Xest[known_good] - X[accepted[known_good, 0]], axis=1)
    # Single variable: change baseline only. Same points and exact same pixel noise.
    pixel_noise = rng.normal(0, .4, (2, n, 2))
    baseline_rows, depth_errors = [], []
    for base in [.005, .02, .08, .32]:
        pr = K @ np.column_stack([np.eye(3), [-base, 0, 0]])
        left = project(X, np.eye(3), np.zeros(3)) + pixel_noise[0]
        right = project(X, np.eye(3), [-base, 0, 0]) + pixel_noise[1]
        xb = triangulate(left, right, P0, pr)
        err = np.abs(xb[:, 2] - X[:, 2])
        depth_errors.append(err)
        reproj = np.linalg.norm(project(xb, np.eye(3), [-base, 0, 0])-right, axis=1)
        baseline_rows.append(dict(baseline_m=base, median_abs_depth_error_m=float(np.median(err)),
            positive_depth_fraction=float(np.mean(xb[:, 2] > 0)),
            median_reprojection_px=float(np.median(reproj))))
    scaled = project(3. * X, R[2], 3. * t[2])
    scale_invariance = float(np.max(np.abs(scaled-uv[2])))
    exact = triangulate(uv[0], uv[1], P0, K @ np.column_stack([R[1], t[1]]))
    exact_error = float(np.max(np.linalg.norm(exact-X, axis=1)))
    # Moving along the stereo baseline can mimic a static point at a false depth.
    p0 = np.array([[.2, .1, 4.]])
    p1 = p0 + [.1, 0, 0]
    u0 = project(p0, np.eye(3), np.zeros(3))
    u1 = project(p1, np.eye(3), [-.4, 0, 0])
    xf = triangulate(u0, u1, P0, K @ np.column_stack([np.eye(3), [-.4, 0, 0]]))
    dynamic_residual = float(np.linalg.norm(project(xf, np.eye(3), [-.4, 0, 0])-u1))
    result = dict(seed=22, scope="synthetic descriptors and geometry, no detector or COLMAP",
        units="world/camera metres; image pixels", points=n, mismatched_identities=54,
        tentative_matches=len(pairs), tentative_true=int(np.sum(pairs[:, 0] == pairs[:, 1])),
        accepted_tracks=len(accepted), accepted_true=int(known_good.sum()),
        accepted_false=int((~known_good).sum()), external_baseline_m=baseline,
        relative_rotation_error_deg=rot_error(Rest, R[1]),
        translation_direction_error_deg=float(np.degrees(np.arccos(np.clip(
            np.dot(test, t[1]) / (np.linalg.norm(test)*np.linalg.norm(t[1])), -1, 1)))),
        triangulated_median_error_m=float(np.median(errors)), pnp_tracks=len(ids),
        pnp_inliers=int(len(pnp_in)), third_rotation_error_deg=rot_error(R2est, R[2]),
        third_camera_center_error_m=float(np.linalg.norm(C2est-C[2])),
        exact_dlt_max_error_m=exact_error, scale3_projection_max_difference_px=scale_invariance,
        baseline_control=baseline_rows, moving_point_true_z_m=4.,
        moving_point_static_estimated_z_m=float(xf[0, 2]),
        moving_point_static_reprojection_px=dynamic_residual)
    assert exact_error < 1e-10 and scale_invariance < 1e-10
    assert result["accepted_false"] <= 2 and result["accepted_true"] >= 100
    np.savez(OUT / "arrays.npz", truth=X, estimated=Xest, accepted=accepted,
             uv=noisy, pairs=pairs, keep=keep, cameras=C,
             baseline_errors=np.array(depth_errors), moving_reconstruction=xf)
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
