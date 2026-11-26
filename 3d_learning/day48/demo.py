"""Day48: planar 3R IK, capsule collision and pose-error propagation. CPU only."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = np.array([0.35, 0.30, 0.12])  # metres
LIMIT = np.deg2rad([170, 150, 170])
Q0 = np.deg2rad([-70, 100, -30])
OBSTACLE = np.array([0.34, 0.08])
RADIUS, LINK_RADIUS = 0.055, 0.012


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def fk(q):
    angles = np.cumsum(q)
    return np.vstack([np.zeros(2), np.cumsum(
        L[:, None] * np.column_stack([np.cos(angles), np.sin(angles)]), axis=0)])


def ik(pose):
    x, y, phi = pose
    wrist = np.array([x, y]) - L[2] * np.array([np.cos(phi), np.sin(phi)])
    c2 = (wrist @ wrist - L[0]**2 - L[1]**2) / (2 * L[0] * L[1])
    if abs(c2) > 1 + 1e-10:
        return []
    out = []
    for sign in [1, -1]:
        q2 = sign * np.arccos(np.clip(c2, -1, 1))
        q1 = np.arctan2(wrist[1], wrist[0]) - np.arctan2(
            L[1] * np.sin(q2), L[0] + L[1] * np.cos(q2))
        q = wrap([q1, q2, phi - q1 - q2])
        if not any(np.allclose(q, old) for old in out):
            out.append(q)
    return out


def point_segment(p, a, b):
    v = b - a
    t = np.clip((p - a) @ v / max(v @ v, 1e-15), 0, 1)
    return np.linalg.norm(p - (a + t * v))


def segment_distance(a, b, c, d):
    v, w = b - a, d - c
    cross = lambda u, z: u[0] * z[1] - u[1] * z[0]
    den = cross(v, w)
    if abs(den) > 1e-12:
        t, u = cross(c - a, w) / den, cross(c - a, v) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return min(point_segment(a, c, d), point_segment(b, c, d),
               point_segment(c, a, b), point_segment(d, a, b))


def clearance(q, margin=0.0):
    p = fk(q)
    env = min(point_segment(OBSTACLE, p[i], p[i+1]) for i in range(3))
    self_distance = segment_distance(p[0], p[1], p[2], p[3])
    return min(env - RADIUS - LINK_RADIUS - margin,
               self_distance - 2 * LINK_RADIUS - margin)


def valid(q, margin=0.0):
    return bool(np.all(np.abs(q) <= LIMIT) and clearance(q, margin) > 0)


def path_to(q, samples=61):
    # Interpolate actual limited joint coordinates, not wrapped shortest arcs.
    return Q0 + np.linspace(0, 1, samples)[:, None] * (q - Q0)


def filter_pose(pose, margin=0.0):
    branches = ik(pose)
    endpoint = [q for q in branches if valid(q, margin)]
    paths = [q for q in endpoint if all(valid(z, margin) for z in path_to(q))]
    best = min(paths, key=lambda q: np.linalg.norm(q-Q0)) if paths else None
    return len(branches), len(endpoint), best


def target(object_pose):
    x, y, theta = object_pose
    # A deliberately simplified tool transform: local offset (-0.10, 0), same yaw.
    return np.array([x - 0.10 * np.cos(theta), y - 0.10 * np.sin(theta), theta])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--margin-mm", type=float, default=0)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day48/results.json")
    args = ap.parse_args()
    assert args.trials > 0 and args.margin_mm >= 0
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    assert valid(Q0), "Start must be collision free"
    # Find a reproducible endpoint-clear but straight-joint-path-blocked example.
    counter = None
    for _ in range(1200):
        q = rng.uniform(-0.95 * LIMIT, 0.95 * LIMIT)
        if valid(q) and any(not valid(z) for z in path_to(q)):
            counter = q
            break
    assert counter is not None
    truth = np.array([0.65, 0.15, 0.2])
    true_target = target(truth)
    branches, endpoints, nominal_q = filter_pose(true_target)
    assert nominal_q is not None, "Nominal grasp proxy needs a feasible path"
    assert np.linalg.norm(fk(nominal_q)[-1] - true_target[:2]) < 1e-10
    assert abs(wrap(nominal_q.sum() - true_target[2])) < 1e-10
    standardized = rng.normal(size=(args.trials, 3))  # paired noise across conditions
    rows = []
    for mm, deg in [(0, 0), (2, 1), (5, 3), (10, 5)]:
        counts = dict(ik=0, endpoint=0, path=0, capture_proxy=0)
        errors = []
        for z in standardized:
            noisy = truth + z * [mm/1000, mm/1000, np.deg2rad(deg)]
            estimated_target = target(noisy)
            ni, ne, q = filter_pose(estimated_target, args.margin_mm/1000)
            counts["ik"] += ni > 0
            counts["endpoint"] += ne > 0
            counts["path"] += q is not None
            error = np.linalg.norm(estimated_target[:2] - true_target[:2])
            errors.append(error)
            counts["capture_proxy"] += bool(q is not None and error <= 0.008
                and abs(wrap(estimated_target[2]-true_target[2])) <= np.deg2rad(8))
        rows.append(dict(translation_std_mm=mm, yaw_std_deg=deg, trials=args.trials,
                         mean_target_error_mm=float(1000*np.mean(errors)), **counts))
    report = dict(python=platform.python_version(), numpy=np.__version__,
        matplotlib=matplotlib.__version__, seed=args.seed, trials=args.trials,
        device="CPU", data="synthetic planar 3R, known circle obstacle",
        unit="m, rad", frame="planar base x-y; object local grasp offset",
        timestamp="static scene, path interpolation parameter is not seconds",
        start_q_rad=Q0.tolist(), link_lengths_m=L.tolist(), margin_mm=args.margin_mm,
        path_samples=61, training_samples=0, real_robot_interactions=0,
        nominal_branches=branches, nominal_clear_endpoints=endpoints,
        counterexample_q_rad=counter.tolist(), rows=rows,
        elapsed_s=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if args.output.stem != "results":
        return
    asset = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day48"
    asset.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        ax.add_patch(plt.Circle(OBSTACLE, RADIUS, color="tomato", alpha=.7))
        ax.set(xlim=(-.45, .85), ylim=(-.6, .65), xlabel="base x (m)", ylabel="base y (m)")
        ax.set_aspect("equal"); ax.grid(alpha=.2)
    p = fk(Q0); axes[0].plot(*p.T, "o--", label="initial")
    for i, q in enumerate(ik(true_target)):
        p = fk(q); axes[0].plot(*p.T, "o-", label=f"IK branch {i+1}, clear={valid(q)}")
    axes[0].set_title("Same tool pose, different link geometry"); axes[0].legend(fontsize=8)
    trajectory = path_to(counter)
    for q in trajectory[::5]:
        p = fk(q); axes[1].plot(*p.T, color="tomato" if not valid(q) else "steelblue", alpha=.4)
    p = fk(counter); axes[1].plot(*p.T, "ko-", label="collision-free endpoint")
    axes[1].set_title("Clear endpoint does not imply clear path"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(asset/"ik_and_path.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rows))
    for name, label in [("endpoint", "endpoint feasible"), ("path", "sampled path feasible"),
                        ("capture_proxy", "capture geometry proxy")]:
        ax.plot(x, [r[name]/args.trials for r in rows], "o-", label=label)
    ax.set(xticks=x, xticklabels=[f'{r["translation_std_mm"]} mm / {r["yaw_std_deg"]} deg' for r in rows],
           ylim=(-.03, 1.05), ylabel="fraction of all attempts", xlabel="pose noise standard deviation")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(asset/"pose_noise.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
