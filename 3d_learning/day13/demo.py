"""D13: CPU synthetic geometry and action-contract experiments.

Requires NumPy; OpenCV is optional for an independent DLT/PnP check.
No camera, network, model checkpoint, ROS node, or robot is accessed.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def project(P, xyz):
    """P: (3,4), xyz: (N,3), result: ideal pixels (N,2)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    q = np.c_[xyz, np.ones(len(xyz))] @ P.T
    if not np.isfinite(q).all() or np.any(q[:, 2] <= 1e-9):
        raise ValueError("Projection requires finite positive camera depth")
    return q[:, :2] / q[:, 2:]


def triangulate(P1, P2, uv1, uv2):
    """Homogeneous DLT, then cheirality checks in both cameras."""
    A = np.stack([
        uv1[:, 0, None] * P1[2] - P1[0],
        uv1[:, 1, None] * P1[2] - P1[1],
        uv2[:, 0, None] * P2[2] - P2[0],
        uv2[:, 1, None] * P2[2] - P2[1],
    ], axis=1)  # (N,4,4)
    _, _, vt = np.linalg.svd(A)
    h = vt[:, -1, :]
    if np.any(np.abs(h[:, 3]) < 1e-10):
        raise ValueError("Point at/near infinity; do not divide by tiny w")
    xyz = h[:, :3] / h[:, 3:]
    project(P1, xyz)
    project(P2, xyz)
    return xyz


def rms_pixels(P1, P2, xyz, uv1, uv2):
    # RMS Euclidean pixel distance per observed 2D point, NOT per coordinate.
    e = np.concatenate([project(P1, xyz) - uv1, project(P2, xyz) - uv2])
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def stereo_cameras(f, baseline):
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.]], float)
    P1 = K @ np.c_[np.eye(3), np.zeros(3)]
    P2 = K @ np.c_[np.eye(3), [-baseline, 0, 0]]
    return K, P1, P2


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def action_endpoint(a, mode, unit, frame, dt, R_base_tool):
    """Translation only, fixed tool orientation, one ideal control interval."""
    a = np.asarray(a, float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("Expected a finite three-vector")
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    scales = {
        ("delta_position", "m"): 1.,
        ("delta_position", "mm"): .001,
        ("linear_velocity", "m/s"): dt,
        ("linear_velocity", "mm/s"): .001 * dt,
    }
    if (mode, unit) not in scales or frame not in ("base", "tool"):
        raise ValueError("Unknown mode/unit/frame contract")
    delta = a * scales[(mode, unit)]
    return R_base_tool @ delta if frame == "tool" else delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--pixel-sigma", type=float, default=.25)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=(
        Path(__file__).resolve().parents[2] /
        "_assets/3d_learning/day13/results.json"))
    args = ap.parse_args()
    if args.pixel_sigma <= 0 or args.samples < 100:
        ap.error("pixel-sigma must be positive; samples must be >= 100")
    rng = np.random.default_rng(args.seed)
    f, B = 600., .06
    K, P1, P2 = stereo_cameras(f, B)
    xyz = rng.uniform([-.15, -.10, .6], [.15, .10, 1.5], (100, 3))
    uv1, uv2 = project(P1, xyz), project(P2, xyz)
    exact = triangulate(P1, P2, uv1, uv2)
    exact_error = float(np.max(np.linalg.norm(exact - xyz, axis=1)))
    assert exact_error < 1e-10

    # Deliberately wrong metric calibration: same correspondences, 5% larger B.
    _, _, P2_wrong = stereo_cameras(f, B * 1.05)
    wrong = triangulate(P1, P2_wrong, uv1, uv2)
    np.testing.assert_allclose(wrong, xyz * 1.05, atol=1e-10)
    geometry = {
        "point_count": len(xyz),
        "clean_max_3d_error_m": exact_error,
        "clean_reprojection_rms_px": rms_pixels(P1, P2, exact, uv1, uv2),
        "wrong_baseline_relative_scale_error": .05,
        "wrong_baseline_3d_rmse_m": float(np.sqrt(np.mean(
            np.sum((wrong - xyz) ** 2, axis=1)))),
        "wrong_baseline_fitted_reprojection_rms_px": rms_pixels(
            P1, P2_wrong, wrong, uv1, uv2),
        "wrong_baseline_true_camera_reprojection_rms_px": rms_pixels(
            P1, P2, wrong, uv1, uv2),
    }

    # Same standardized random errors in every depth/baseline condition.
    eps = rng.normal(size=(args.samples, 2))
    stereo = []
    sigma_d = np.sqrt(2) * args.pixel_sigma
    for b in [.06, .12]:
        for z in [.5, 1., 2.]:
            d_true = f * b / z
            d_obs = d_true + args.pixel_sigma * (eps[:, 0] - eps[:, 1])
            valid = np.isfinite(d_obs) & (d_obs > 1e-6)
            z_hat = f * b / d_obs[valid]
            if not len(z_hat):
                raise ValueError("No valid disparities in this condition")
            e = z_hat - z
            predicted = z * z / (f * b) * sigma_d
            stereo.append({
                "baseline_m": b, "depth_m": z, "disparity_px": d_true,
                "trials": args.samples, "valid": int(valid.sum()),
                "sigma_z_linear_m": float(predicted),
                "sigma_z_empirical_m": float(np.std(z_hat, ddof=1)),
                "bias_m": float(np.mean(e)),
                "rmse_m": float(np.sqrt(np.mean(e * e))),
                "p95_absolute_error_m": float(np.quantile(abs(e), .95)),
                "within_20mm": int(np.sum(abs(e) < .02)),
            })

    # Correlation matters because d = u_L - u_R shares u_L.
    z, u, v = 1., 440., 300.
    d = f * B / z
    J = np.array([[B/d, 0, -B*(u-320)/d**2],
                  [0, B/d, -B*(v-240)/d**2],
                  [0, 0, -f*B/d**2]])
    s2 = args.pixel_sigma ** 2
    cov_uvd = s2 * np.array([[1, 0, 1], [0, 1, 0], [1, 0, 2.]])
    cov_xyz = J @ cov_uvd @ J.T
    naive_cov = J @ np.diag(np.diag(cov_uvd)) @ J.T
    eps3 = rng.normal(size=(args.samples, 3)) * args.pixel_sigma
    uu, vv = u + eps3[:, 0], v + eps3[:, 1]
    dd = d + eps3[:, 0] - eps3[:, 2]
    valid = dd > 1e-6
    cloud = np.c_[(uu[valid]-320)*B/dd[valid],
                  (vv[valid]-240)*B/dd[valid], f*B/dd[valid]]
    uncertainty = {
        "input_order": ["u_L_px", "v_L_px", "disparity_px"],
        "xyz_linear_std_m": np.sqrt(np.diag(cov_xyz)).tolist(),
        "xyz_naive_independent_std_m": np.sqrt(np.diag(naive_cov)).tolist(),
        "xyz_empirical_std_m": np.std(cloud, axis=0, ddof=1).tolist(),
        "valid": int(valid.sum()), "trials": args.samples,
        "xz_error_samples_m": (cloud[:600, [0, 2]] - [.2, 1.]).tolist(),
        "cov_xz_linear_m2": cov_xyz[np.ix_([0, 2], [0, 2])].tolist(),
    }

    # Eye-in-hand timing example: fixed point in base, translating camera.
    true_base = np.array([.5, 0., .8])
    p_cam_at_capture = np.array([.1, 0., .8])
    camera_at_capture = np.array([.4, 0., 0.])
    camera_velocity = np.array([.1, 0., 0.])  # m/s, synthetic
    stale = []
    for age in [0., .05, .10, .20]:
        using_latest = p_cam_at_capture + camera_at_capture + camera_velocity*age
        stale.append({"age_s": age,
                      "position_error_m": float(np.linalg.norm(using_latest-true_base))})

    # Fixed tool yaw +90deg. Desired base-frame displacement is +20mm in x.
    R = rz(np.pi / 2)
    target = np.array([.02, 0., 0.])
    local = R.T @ target
    commands = [
        ("correct_base_delta", target, "delta_position", "m", "base", .1),
        ("correct_tool_delta", local, "delta_position", "m", "tool", .1),
        ("wrong_frame", target, "delta_position", "m", "tool", .1),
        ("wrong_mm_as_m", target*1000, "delta_position", "m", "base", .1),
        ("wrong_delta_as_velocity", target, "linear_velocity", "m/s", "base", .1),
        ("correct_velocity", target/.1, "linear_velocity", "m/s", "base", .1),
    ]
    actions = []
    for name, a, mode, unit, frame, dt in commands:
        endpoint = action_endpoint(a, mode, unit, frame, dt, R)
        error = float(np.linalg.norm(endpoint - target))
        actions.append({"case": name, "endpoint_m": endpoint.tolist(),
                        "error_m": error, "within_1mm": bool(error < .001)})
    assert sum(a["within_1mm"] for a in actions) == 3
    frequency = []
    for hz in [10, 20, 50]:
        velocity_displacement = hz * .02 * (1/hz)
        repeated_delta_displacement = hz * .002
        frequency.append({"hz": hz, "duration_s": 1.,
                          "velocity_endpoint_m": velocity_displacement,
                          "fixed_delta_endpoint_m": repeated_delta_displacement})
    np.testing.assert_allclose([a["velocity_endpoint_m"] for a in frequency], .02)

    # Guardrails are data-contract checks, not physical safety certification.
    invalid_rejected = 0
    for mode, unit, frame, dt in [
        ("delta_position", "m/s", "base", .1),
        ("linear_velocity", "m/s", "camera", .1),
        ("delta_position", "m", "base", 0.),
    ]:
        try:
            action_endpoint(target, mode, unit, frame, dt, R)
        except ValueError:
            invalid_rejected += 1
    assert invalid_rejected == 3
    start = time.perf_counter()
    for _ in range(50):
        triangulate(P1, P2, uv1, uv2)
    latency_ms = (time.perf_counter() - start) * 1000 / 50

    cv_result = {"status": "not_run", "reason": "opencv-python not installed"}
    try:
        import cv2
    except ImportError:
        pass
    else:
        h = cv2.triangulatePoints(P1, P2, uv1.T, uv2.T)
        cv_xyz = (h[:3] / h[3]).T
        cv_diff = float(np.max(np.linalg.norm(cv_xyz - exact, axis=1)))
        assert cv_diff < 1e-10
        obj = rng.uniform(-.05, .05, (30, 3)).astype(np.float64)
        rvec = np.array([.1, -.15, .05], float)
        tvec = np.array([.05, -.03, .8], float)
        pixels, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
        pnp = []
        for scale in [1., 1000.]:
            ok, r_hat, t_hat = cv2.solvePnP(
                obj*scale, pixels, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
            assert ok
            fitted, _ = cv2.projectPoints(obj*scale, r_hat, t_hat, K, np.zeros(5))
            err_px = float(np.sqrt(np.mean(np.sum(
                (fitted.reshape(-1, 2)-pixels.reshape(-1, 2))**2, axis=1))))
            assert err_px < 1e-6
            np.testing.assert_allclose(t_hat.ravel()/scale, tvec, atol=1e-7)
            pnp.append({"object_coordinate_scale": scale,
                        "translation_in_input_length_unit": t_hat.ravel().tolist(),
                        "reprojection_rms_px": err_px})
        cv_result = {"status": "passed", "version": cv2.__version__,
                     "dlt_max_difference_m": cv_diff, "pnp_scale_cases": pnp}

    report = {
        "metadata": {
            "data": "synthetic ideal rectified stereo; no distortion or hardware",
            "modalities": ["ideal image coordinates", "synthetic actions"],
            "python": platform.python_version(), "numpy": np.__version__,
            "device": "CPU", "seed": args.seed, "pixel_sigma_px": args.pixel_sigma,
            "length_unit": "m", "camera_axes": "x right, y down, z forward",
            "timing": "synthetic seconds; each action dt explicit; no sensor clock",
            "initial_action_position_m": [0., 0., 0.],
            "initial_tool_yaw_rad": float(np.pi/2),
            "robot_interactions": 0, "training_steps": 0,
            "dlt_100_points_mean_ms_50_repeats": latency_ms,
            "latency_scope": "local DLT + validity checks, no IO/perception/control",
        },
        "geometry": geometry, "stereo_noise": stereo,
        "uncertainty": uncertainty, "timestamp_cases": stale,
        "action_cases": actions, "frequency_cases": frequency,
        "invalid_contract_rejections": {"rejected": invalid_rejected, "attempts": 3},
        "opencv_check": cv_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "geometry": geometry,
                      "action_examples_within_1mm": "3/6 deliberately mixed cases",
                      "opencv": cv_result["status"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
