"""Day19: Sim(3), robust correspondences, ICP, symmetry, and action frames.

CPU synthetic geometry only. Row arrays implement y = s * (x @ R.T) + t.
x is dimensionless centered NOCS-like coordinates (NOCS - 0.5), not network
output. R maps object to camera. Scale is metres per canonical coordinate unit.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def apply(x, scale, rotation, translation):
    return scale * (x @ rotation.T) + translation


def align(x, y, estimate_scale=True):
    """Umeyama: H=mean(y_centered x_centered.T), H=U diag(d) V.T."""
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    if len(x) < 3 or np.linalg.matrix_rank(xc, tol=1e-10) < 2:
        raise ValueError("Need at least three non-collinear source points")
    h = yc.T @ xc / len(x)
    u, d, vt = np.linalg.svd(h)
    sign = np.ones(3)
    sign[-1] = 1. if np.linalg.det(u @ vt) >= 0 else -1.
    rotation = (u * sign) @ vt
    variance = np.mean(np.sum(xc * xc, axis=1))
    scale = float(np.sum(d * sign) / variance) if estimate_scale else 1.
    if scale <= 0:
        raise ValueError("Positive object scale required")
    translation = y.mean(axis=0) - scale * rotation @ x.mean(axis=0)
    return scale, rotation, translation


def robust_align(x, y, rng, threshold=.003, trials=300):
    best_mask, best_score = None, (-1, -np.inf)
    for _ in range(trials):
        ids = rng.choice(len(x), 3, replace=False)
        try:
            fit = align(x[ids], y[ids])
        except ValueError:
            continue
        residual = np.linalg.norm(apply(x, *fit) - y, axis=1)
        mask = residual < threshold
        if mask.sum() < 3:
            continue
        score = (int(mask.sum()), -float(np.median(residual[mask])))
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None:
        raise RuntimeError("No valid RANSAC consensus")
    for _ in range(5):
        fit = align(x[best_mask], y[best_mask])
        mask = np.linalg.norm(apply(x, *fit) - y, axis=1) < threshold
        if np.array_equal(mask, best_mask):
            break
        if mask.sum() < 3:
            raise RuntimeError("Consensus lost during refinement")
        best_mask = mask
    fit = align(x[best_mask], y[best_mask])
    final_mask = np.linalg.norm(apply(x, *fit) - y, axis=1) < threshold
    return fit, final_mask


def nearest(x, y):
    """Brute-force nearest neighbours; suitable only for these small clouds."""
    squared = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    ids = np.argmin(squared, axis=1)
    return ids, np.sqrt(squared[np.arange(len(x)), ids])


def icp(x_metric, y_metric, rotation, translation, iterations=80):
    """Point-to-point ICP, fixed scale 1; no outlier gate or visibility model."""
    rotation, translation = rotation.copy(), translation.copy()
    history = []
    for _ in range(iterations):
        transformed = apply(x_metric, 1., rotation, translation)
        ids, distances = nearest(transformed, y_metric)
        history.append(float(np.sqrt(np.mean(distances ** 2))))
        _, dr, dt = align(transformed, y_metric[ids], estimate_scale=False)
        rotation, translation = dr @ rotation, dr @ translation + dt
        if np.linalg.norm(dr - np.eye(3)) + np.linalg.norm(dt) < 1e-10:
            break
    _, distances = nearest(apply(x_metric, 1., rotation, translation), y_metric)
    history.append(float(np.sqrt(np.mean(distances ** 2))))
    return rotation, translation, np.array(history)


def rotation_error_deg(estimated, truth):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(estimated @ truth.T)-1)/2, -1, 1))))


def rotation_from_6d(columns, eps=1e-10):
    """Zhou-style two-column convention; reject zero/collinear input."""
    first, second = columns[:3], columns[3:]
    if np.linalg.norm(first) < eps:
        raise ValueError("First rotation column is zero")
    b1 = first / np.linalg.norm(first)
    projected = second - (b1 @ second) * b1
    if np.linalg.norm(projected) < eps:
        raise ValueError("Rotation columns are collinear")
    b2 = projected / np.linalg.norm(projected)
    return np.column_stack((b1, b2, np.cross(b1, b2)))


def pose(rotation, translation):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, translation
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "_assets/3d_learning/day19")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    begin = time.perf_counter()
    # Interior samples of a notched box; partial asymmetry, not a mesh surface.
    candidates = rng.uniform([-.42, -.24, -.16], [.42, .24, .16], (400, 3))
    raw_shape = candidates[~((candidates[:, 0] > .12) & (candidates[:, 1] > .02))][:180]
    bbox_min, bbox_max = raw_shape.min(axis=0), raw_shape.max(axis=0)
    # One common normalization scale: tight canonical bounding-box diagonal = 1.
    canonical = (raw_shape - (bbox_min+bbox_max)/2) / np.linalg.norm(bbox_max-bbox_min)
    assert np.isclose(np.linalg.norm(np.ptp(canonical, axis=0)), 1.)
    scale_true = .32
    rotation_true = rz(np.deg2rad(35.)) @ ry(np.deg2rad(-20.))
    translation_true = np.array([.10, -.04, .80])
    camera_clean = apply(canonical, scale_true, rotation_true, translation_true)
    exact = align(canonical, camera_clean)
    assert abs(exact[0] - scale_true) < 1e-12
    np.testing.assert_allclose(exact[1], rotation_true, atol=1e-12)
    np.testing.assert_allclose(exact[2], translation_true, atol=1e-12)
    camera_noisy = camera_clean + rng.normal(0., .0006, camera_clean.shape)
    contaminated = camera_noisy.copy()
    bad_ids = rng.choice(len(canonical), 54, replace=False)
    contaminated[bad_ids] = camera_noisy[np.roll(bad_ids, 1)]
    known_inlier = np.ones(len(canonical), dtype=bool)
    known_inlier[bad_ids] = False
    least_squares = align(canonical, contaminated)
    robust, mask = robust_align(canonical, contaminated, rng)
    alignment = {}
    for name, fit in [("ordinary_ls", least_squares), ("ransac_refit", robust)]:
        pred = apply(canonical, *fit)
        alignment[name] = {
            "scale_m_per_unit": fit[0], "scale_relative_error_percent": 100*abs(fit[0]/scale_true-1),
            "rotation_error_deg": rotation_error_deg(fit[1], rotation_true),
            "translation_error_m": float(np.linalg.norm(fit[2]-translation_true)),
            "clean_correspondence_rmse_m": float(np.sqrt(np.mean(np.sum((pred-camera_clean)**2, axis=1)))),
            "observed_correspondence_rmse_m": float(np.sqrt(np.mean(np.sum((pred-contaminated)**2, axis=1)))),
        }
    assert np.sum(mask & ~known_inlier) == 0
    assert np.sum(mask & known_inlier) >= 120
    assert alignment["ransac_refit"]["clean_correspondence_rmse_m"] < .001
    source_metric = scale_true * canonical
    r_good = rotation_true @ rz(np.deg2rad(8.)) @ ry(np.deg2rad(-4.))
    r_bad = rotation_true @ rz(np.pi)
    t_bad = camera_noisy.mean(axis=0) - r_bad @ source_metric.mean(axis=0)
    icp_records, icp_arrays = {}, {}
    for name, r0, t0 in [("good_initialization", r_good, translation_true + [.008, -.005, .006]),
                         ("half_turn_initialization", r_bad, t_bad)]:
        r, t, history = icp(source_metric, camera_noisy, r0, t0)
        pred = apply(source_metric, 1., r, t)
        icp_records[name] = {
            "updates": len(history)-1, "nn_rmse_m": float(history[-1]),
            "rotation_error_deg": rotation_error_deg(r, rotation_true),
            "translation_error_m": float(np.linalg.norm(t-translation_true)),
            "known_correspondence_rmse_m": float(np.sqrt(np.mean(np.sum((pred-camera_clean)**2, axis=1)))),
        }
        icp_arrays[name+"_history"] = history
        icp_arrays[name+"_points"] = pred
    assert icp_records["good_initialization"]["known_correspondence_rmse_m"] < .001
    # Separate exact-symmetry example: 72 angular samples x 9 height samples.
    theta, z = np.meshgrid(np.arange(72)*2*np.pi/72, np.linspace(-.06, .06, 9))
    cylinder = np.c_[.05*np.cos(theta.ravel()), .05*np.sin(theta.ravel()), z.ravel()]
    cylinder_rotated = cylinder @ rz(np.pi).T
    add = float(np.mean(np.linalg.norm(cylinder_rotated-cylinder, axis=1)))
    _, sym_distances = nearest(cylinder, cylinder_rotated)  # Ground truth -> estimate.
    adds = float(sym_distances.mean())
    assert np.isclose(add, .10) and adds < 1e-12
    current = pose(rz(np.pi/2), [.4, .1, .3])
    local_delta = np.array([.02, 0., 0.])
    tool_increment = pose(np.eye(3), local_delta)
    correct = current @ tool_increment
    base_delta = current[:3, :3] @ local_delta
    equivalent = pose(np.eye(3), base_delta) @ current
    wrong_left = tool_increment @ current
    wrong_point_as_delta = current[:3, 3] + current[:3, :3] @ local_delta + current[:3, 3]
    rotation_delta = pose(rz(np.pi/2), np.zeros(3))
    left_rotated = rotation_delta @ current  # Rotate the whole pose about base origin.
    around_tcp = pose(rotation_delta[:3, :3] @ current[:3, :3], current[:3, 3])
    np.testing.assert_allclose(correct, equivalent, atol=1e-12)
    np.testing.assert_allclose(correct[:3, 3], [.4, .12, .3], atol=1e-12)
    sixd = np.r_[rotation_true[:, 0], rotation_true[:, 1]]
    rebuilt = rotation_from_6d(sixd)
    np.testing.assert_allclose(rebuilt, rotation_true, atol=1e-12)
    rejected = 0
    for degenerate in [np.zeros(6), np.array([1.,0.,0.,2.,0.,0.])]:
        try:
            rotation_from_6d(degenerate)
        except ValueError:
            rejected += 1
    assert rejected == 2
    report = {
        "metadata": {"verified_date": "2026-09-23", "seed": args.seed,
                     "python": platform.python_version(), "numpy": np.__version__, "device": "CPU",
                     "elapsed_ms": 1000*(time.perf_counter()-begin), "training_steps": 0, "robot_interactions": 0},
        "data": {"N": len(canonical), "coordinate_definition": "centered NOCS x = nocs - 0.5; bounding-box center=0 and diagonal=1, dimensionless",
                 "scale_true_m_per_canonical_unit": scale_true, "R_camera_object": rotation_true.tolist(),
                 "t_camera_object_m": translation_true.tolist(), "coordinate_noise_std_m": .0006,
                 "permuted_correspondences": len(bad_ids), "ransac_threshold_m": .003, "ransac_trials": 300},
        "exact_alignment_max_error_m": float(np.max(np.linalg.norm(apply(canonical, *exact)-camera_clean, axis=1))),
        "alignment": alignment,
        "ransac_consensus": {"count": int(mask.sum()), "true_positives": int(np.sum(mask & known_inlier)),
                             "false_positives": int(np.sum(mask & ~known_inlier)), "true_inliers": int(known_inlier.sum())},
        "icp": icp_records,
        "symmetry": {"points": len(cylinder), "radius_m": .05, "height_m": .12,
                     "rotation_difference_deg": 180., "axis": "object z (teaching convention, not NOCS dataset symmetry-axis convention)",
                     "ADD_m": add, "ADD_S_m": adds, "ADD_S_direction": "ground-truth points -> estimated surface, one-sided mean",
                     "tracked_semantic_point_error_m": float(np.linalg.norm(cylinder[0]-cylinder_rotated[0])),
                     "scope": "Unlabelled discrete cylindrical side samples; no handle, label, grasp or tool geometry"},
        "action": {"current_T_base_tcp": current.tolist(), "tool_delta_m": local_delta.tolist(),
                   "base_delta_m": base_delta.tolist(), "absolute_base_target_m": correct[:3, 3].tolist(),
                   "wrong_left_translation_target_m": wrong_left[:3, 3].tolist(),
                   "wrong_left_translation_error_m": float(np.linalg.norm(wrong_left[:3, 3]-correct[:3, 3])),
                   "wrong_point_as_delta_target_m": wrong_point_as_delta.tolist(),
                   "wrong_point_as_delta_error_m": float(np.linalg.norm(wrong_point_as_delta-correct[:3, 3])),
                   "pure_left_rotation_position_m": left_rotated[:3, 3].tolist(),
                   "rotate_about_current_tcp_position_m": around_tcp[:3, 3].tolist(),
                   "rotation_pivot_position_difference_m": float(np.linalg.norm(left_rotated[:3, 3]-around_tcp[:3, 3]))},
        "rotation_6d": {"convention": "concatenated first two COLUMNS of R", "degenerate_inputs_rejected": rejected,
                        "roundtrip_max_entry_error": float(np.max(np.abs(rebuilt-rotation_true)))},
        "limits": ["No NOCS network: correspondence oracle plus artificial mismatches",
                   "RANSAC assumes one isotropic positive scale and a rigid object",
                   "ICP uses full clouds, known fixed metric scale and nearest neighbours; no occlusion or outlier rejection",
                   "Nearest-neighbour fit quality is separate from pose error; symmetry can make the pose unidentifiable",
                   "No robot, collision, kinematic or dynamic validation of action examples"],
    }
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(args.out_dir / "arrays.npz", canonical=canonical, camera_clean=camera_clean,
        contaminated=contaminated, known_inlier=known_inlier, ransac_mask=mask,
        ordinary_ls_points=apply(canonical, *least_squares), ransac_points=apply(canonical, *robust),
        cylinder=cylinder, cylinder_rotated=cylinder_rotated, **icp_arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
