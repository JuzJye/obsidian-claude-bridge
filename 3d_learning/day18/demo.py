"""D18 CPU laboratory: RGB-D/BEV, depth ambiguity, fusion, IK and trajectories.

NumPy only. All inputs are synthetic. No robot, ROS, network or model is used.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


LENGTHS = np.array([.5, .4])
LINK_RADIUS = .02
OBSTACLES = np.array([[.28, .25, .09], [.70, -.40, .06]])


def scene():
    """Ideal top surfaces: table, cup top and two obstacle tops; optical Z."""
    h, w, f, cx, cy = 96, 120, 80., 60., 48.
    v, u = np.mgrid[:h, :w]
    depth = np.full((h, w), 1.2)
    label = np.zeros((h, w), dtype=np.int64)
    objects = [(.60, .20, .06, .20, 1), (.28, .25, .09, .35, 2),
               (.70, -.40, .06, .30, 2)]
    for x, y, radius, height, category in objects:
        z = 1.2 - height
        bx, by = .4 + (u-cx)*z/f, -(v-cy)*z/f
        hit = ((bx-x)**2+(by-y)**2 <= radius**2) & (z < depth)
        depth[hit], label[hit] = z, category
    depth[60:70, 40:50] = np.nan  # unobserved patch; never silently set to zero
    rgb = np.array([[190, 201, 210], [35, 145, 210], [235, 145, 55]])[label]
    rgb[~np.isfinite(depth)] = [40, 40, 40]
    return depth, label, rgb.astype(np.uint8)


def unproject(depth):
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    cam = np.c_[(u[valid]-60)*z/80., (v[valid]-48)*z/80., z]
    # T_base_camera: R=diag(1,-1,-1), t=(0.4,0,1.2), all lengths in m.
    base = cam @ np.diag([1., -1., -1.]).T + [.4, 0., 1.2]
    return base, valid


def bev(points, labels, resolution=.02):
    """Count/max-height/semantic-ratio; zero count means no sample, NOT free."""
    lo = np.array([-.4, -.7])
    hi = np.array([1.2, .7])
    n = np.rint((hi-lo)/resolution).astype(int)  # x,y dimensions
    inside = np.all((points[:, :2] >= lo) & (points[:, :2] < hi), axis=1)
    p, lab = points[inside], labels[inside]
    ij = np.floor((p[:, :2]-lo)/resolution).astype(int)
    # Explicitly reject boundaries; do not clip off-grid evidence into edge cells.
    keep = np.all((ij >= 0) & (ij < n), axis=1)
    p, lab, ij = p[keep], lab[keep], ij[keep]
    count = np.zeros((n[1], n[0]), dtype=np.int64)
    cup_count = np.zeros_like(count)
    height = np.full(count.shape, -np.inf)
    np.add.at(count, (ij[:, 1], ij[:, 0]), 1)
    np.add.at(cup_count, (ij[:, 1], ij[:, 0]), lab == 1)
    np.maximum.at(height, (ij[:, 1], ij[:, 0]), p[:, 2])
    height[count == 0] = np.nan
    fraction = np.divide(cup_count, count, out=np.zeros_like(height), where=count > 0)
    center = lo+(ij+.5)*resolution
    return {
        "height": height, "count": count, "cup_fraction": fraction,
        "xy_extent": np.array([lo[0], hi[0], lo[1], hi[1]]),
        "max_sample_quantization_error_m": float(np.max(
            np.linalg.norm(center-p[:, :2], axis=1))),
        "empty_cells": int(np.sum(count == 0)),
    }


def fk(q):
    elbow = LENGTHS[0]*np.array([np.cos(q[0]), np.sin(q[0])])
    tip = elbow+LENGTHS[1]*np.array([np.cos(q.sum()), np.sin(q.sum())])
    return np.array([[0., 0.], elbow, tip])


def ik(xy):
    c2 = (xy @ xy - LENGTHS @ LENGTHS)/(2*np.prod(LENGTHS))
    if c2 < -1-1e-12 or c2 > 1+1e-12:
        return []
    c2 = np.clip(c2, -1., 1.)  # only numerical roundoff, not unreachable targets
    answers = []
    for sign in [1, -1]:
        q2 = sign*np.arccos(c2)
        q1 = np.arctan2(xy[1], xy[0])-np.arctan2(
            LENGTHS[1]*np.sin(q2), LENGTHS[0]+LENGTHS[1]*np.cos(q2))
        q = np.array([q1, q2])
        if np.all(abs(q) <= 2.8):  # illustrative joint limits in rad
            answers.append(q)
    return answers


def clearance(q):
    """Signed clearance of two planar capsules against known circular obstacles."""
    joints = fk(q)
    distances = []
    for ox, oy, radius in OBSTACLES:
        center = np.array([ox, oy])
        for a, b in zip(joints[:-1], joints[1:]):
            ab = b-a
            fraction = np.clip((center-a) @ ab / (ab @ ab), 0., 1.)
            distances.append(np.linalg.norm(center-a-fraction*ab)-radius-LINK_RADIUS)
    return float(min(distances))


def edge_check(a, b, samples=201):
    configurations = np.linspace(a, b, samples)
    values = np.array([clearance(q) for q in configurations])
    # Any link point moves at most M*ds along this straight joint-space edge.
    dq = abs(b-a)
    M = LENGTHS.sum()*dq[0]+LENGTHS[1]*dq[1]
    bound = float(values.min()-M/(2*(samples-1)))
    return {"sample_min_clearance_m": float(values.min()),
            "continuous_lower_bound_m": bound, "samples": samples,
            "certified_in_toy_model": bool(bound > 0),
            "colliding_samples": int(np.sum(values <= 0))}


def timed_path(waypoints):
    """Cubic time scaling on each straight edge, stopping at each waypoint."""
    vmax, amax = np.array([.8, 1.2]), np.array([1.5, 2.])
    records, segments, offset = [], [], 0.
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dq = b-a
        duration = 1.01*max(float(np.max(1.5*abs(dq)/vmax)),
                            float(np.max(np.sqrt(6*abs(dq)/amax))))
        tau = np.linspace(0, 1, 201)
        s = 3*tau**2-2*tau**3
        sd = (6*tau-6*tau**2)/duration
        sdd = (6-12*tau)/duration**2
        q = a+s[:, None]*dq
        qd, qdd = sd[:, None]*dq, sdd[:, None]*dq
        analytic_v = 1.5*abs(dq)/duration
        analytic_a = 6*abs(dq)/duration**2
        assert np.all(analytic_v <= vmax) and np.all(analytic_a <= amax)
        records.append(np.c_[offset+tau*duration, q, qd, qdd])
        segments.append({"duration_s": duration, "max_speed_rad_s": analytic_v.tolist(),
                         "max_accel_rad_s2": analytic_a.tolist()})
        offset += duration
    # At internal waypoints there are two one-sided acceleration samples.
    return np.concatenate(records), {
        "segments": segments, "total_duration_s": offset,
        "velocity_limits_rad_s": vmax.tolist(), "accel_limits_rad_s2": amax.tolist(),
        "acceleration_continuous_at_waypoints": False,
        "torque_limits_verified": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-scale", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--out-dir", type=Path, default=(
        Path(__file__).resolve().parents[2]/"_assets/3d_learning/day18"))
    args = ap.parse_args()
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        ap.error("depth-scale must be finite and positive")
    rng = np.random.default_rng(args.seed)
    depth, label, rgb = scene()
    xyz, valid = unproject(depth)
    biased, _ = unproject(depth*args.depth_scale)
    labs = label[valid]
    target = xyz[labs == 1].mean(axis=0)
    wrong_target = biased[labs == 1].mean(axis=0)
    np.testing.assert_allclose(target, [.6, .2, .2], atol=1e-12)
    fine, coarse = bev(xyz, labs), bev(xyz, labs, .10)
    bias_bev = bev(biased, labs)
    # Learned-depth ambiguity example: manually specified distribution, no learning.
    depths = np.array([.6, 1.4])
    masses = np.array([.5, .5])
    hypotheses = np.array([.4, 0., 1.2])+depths[:, None]*[.2, .2, -1.]
    mean_point = masses @ hypotheses
    assert np.isclose(masses.sum(), 1.)

    fusion = []
    eps = rng.normal(size=(5000, 2))
    for name, bias, reported_sigma in [
        ("independent_unbiased", 0., .04),
        ("biased_overconfident_second_source", .08, .005),
    ]:
        values = 1.+eps*np.array([.01, .04])+np.array([0., bias])
        precision = 1/np.array([.01, reported_sigma])**2
        weights = precision/precision.sum()
        fused = values @ weights
        fusion.append({"case": name, "weights": weights.tolist(),
                       "trials": len(fused), "actual_second_noise_std_m": .04,
                       "reported_second_std_m": reported_sigma,
                       "bias_m": float(np.mean(fused-1)),
                       "rmse_m": float(np.sqrt(np.mean((fused-1)**2))),
                       "claimed_std_m": float(np.sqrt(1/precision.sum()))})

    begin = time.perf_counter()
    solutions = ik(target[:2])
    assert len(solutions) == 2
    branches = []
    for q in solutions:
        error = float(np.linalg.norm(fk(q)[-1]-target[:2]))
        assert error < 1e-12
        branches.append({"q_rad": q.tolist(), "fk_error_m": error,
                         "clearance_m": clearance(q)})
    assert sum(x["clearance_m"] > 0 for x in branches) == 1
    assert not ik(np.array([1.1, 0.]))
    goal = solutions[0]
    start, via = np.array([-1., -.8]), np.array([-1.6, 1.8])
    direct = edge_check(start, goal)
    detour = [edge_check(start, via), edge_check(via, goal)]
    assert clearance(start) > 0 and clearance(goal) > 0
    assert direct["colliding_samples"] > 0
    assert all(x["certified_in_toy_model"] for x in detour)
    waypoints = np.array([start, via, goal])
    motion, timing = timed_path(waypoints)
    np.testing.assert_allclose(motion[0, 1:3], start, atol=1e-12)
    np.testing.assert_allclose(motion[-1, 1:3], goal, atol=1e-12)
    planning_ms = (time.perf_counter()-begin)*1000
    report = {
        "metadata": {"date": "2026-09-22", "python": platform.python_version(),
            "numpy": np.__version__, "device": "CPU", "seed": args.seed,
            "data": "synthetic optical depth + supplied semantic masks",
            "length_unit": "m", "angle_unit": "rad", "time_unit": "s",
            "capture_time_s": 0., "obstacles_static": True,
            "camera_to_base": {"R": np.diag([1,-1,-1]).tolist(), "t": [.4,0,1.2]},
            "training_steps": 0, "robot_interactions": 0,
            "planning_checks_elapsed_ms": planning_ms,
            "latency_scope": "IK, three edges, cubic samples; no sensor/inference/control"},
        "perception": {"cup_surface_center_m": target.tolist(),
            "biased_surface_center_m": wrong_target.tolist(), "depth_scale": args.depth_scale,
            "center_error_m": float(np.linalg.norm(wrong_target-target)),
            "valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
            "cup_pixels": int(np.sum(labs == 1)),
            "fine_empty_cells": fine["empty_cells"],
            "fine_quantization_max_m": fine["max_sample_quantization_error_m"],
            "coarse_quantization_max_m": coarse["max_sample_quantization_error_m"]},
        "depth_ambiguity": {"depths_m": depths.tolist(), "masses": masses.tolist(),
            "hypotheses_base_m": hypotheses.tolist(), "mean_point_m": mean_point.tolist(),
            "mean_is_a_supported_hypothesis": False},
        "fusion": fusion,
        "planning": {"link_lengths_m": LENGTHS.tolist(), "link_radius_m": LINK_RADIUS,
            "obstacles_x_y_radius_m": OBSTACLES.tolist(), "ik_branches": branches,
            "start_q_rad": start.tolist(), "goal_q_rad": goal.tolist(),
            "via_q_rad": via.tolist(), "direct": direct, "detour": detour,
            "planner": "manually supplied waypoint, not RRT or a complete planner",
            "collision_scope": "planar link capsules vs circles; no self/base/tool/payload check",
            "timing": timing},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.out_dir/"arrays.npz", depth=depth, rgb=rgb, label=label,
        points=xyz, fine_height=fine["height"], count=fine["count"],
        coarse_height=coarse["height"], biased_height=bias_bev["height"],
        extent=fine["xy_extent"], motion=motion, waypoints=waypoints)
    print(json.dumps({"output": str(args.out_dir), "perception": report["perception"],
        "ik_branches": branches, "direct": direct, "detour": detour,
        "timing": timing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
