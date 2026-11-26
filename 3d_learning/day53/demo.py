"""D53: 3D frame contract and numerical IK for a synthetic six-revolute-joint arm."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = np.array([0, .30, .25, 0, 0, 0.])
D = np.array([.20, 0, 0, .10, 0, .08])
ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
LIMIT = np.deg2rad([175, 140, 150, 175, 150, 175])
LENGTH_WEIGHT = .20  # metres per radian in the least-squares task metric


def dh(theta, a, d, alpha):
    c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                     [0, sa, ca, d], [0, 0, 0, 1.]])


def fk(q):
    t = np.eye(4)
    origins, axes = [], []
    points = [np.zeros(3)]
    for angle, a, d, alpha in zip(q, A, D, ALPHA):
        origins.append(t[:3, 3].copy())
        axes.append(t[:3, 2].copy())
        t = t @ dh(angle, a, d, alpha)
        points.append(t[:3, 3].copy())
    jac = np.vstack([np.cross(axes, t[:3, 3]-np.array(origins)).T, np.array(axes).T])
    return t, jac, np.array(points)


def log_rotation(r):
    angle = np.arccos(np.clip((np.trace(r)-1)/2, -1, 1))
    vee = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if angle < 1e-7:
        return .5*vee
    if np.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh((r+r.T)/2)
        axis = vectors[:, -1]
        if axis @ vee < 0:
            axis = -axis
        return angle*axis
    return angle/(2*np.sin(angle))*vee


def pose_error(goal, actual):
    return np.r_[goal[:3, 3]-actual[:3, 3], log_rotation(goal[:3, :3]@actual[:3, :3].T)]


def ik(goal, initial, max_steps=160):
    q = np.asarray(initial, dtype=float).copy()
    history = []
    status = "max_steps"
    scale = np.r_[np.ones(3), np.full(3, LENGTH_WEIGHT)]
    for _ in range(max_steps):
        actual, jac, _ = fk(q)
        e = pose_error(goal, actual)
        history.append([np.linalg.norm(e[:3])*1000, np.rad2deg(np.linalg.norm(e[3:]))])
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
            status = "converged"
            break
        j, ew = scale[:, None]*jac, scale*e
        delta = np.linalg.solve(j.T@j+.01**2*np.eye(6), j.T@ew)
        delta *= min(1., .20/max(np.linalg.norm(delta), 1e-12))
        accepted = False
        for step in [1., .5, .25, .125, .0625, .03125]:
            candidate = q+step*delta
            if np.any(np.abs(candidate) > LIMIT):
                continue
            trial = pose_error(goal, fk(candidate)[0])*scale
            if trial@trial < ew@ew:
                q, accepted = candidate, True
                break
        if not accepted:
            status = "stalled_or_joint_limit"
            break
    # Report final state even when the last iteration did not enter the stop branch.
    e = pose_error(goal, fk(q)[0])
    ok = bool(np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4)
    return q, np.array(history), "converged" if ok else status


def make_t(r, p):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-length-mm", type=float, default=120.)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day53")
    args = ap.parse_args()
    if args.tool_length_mm < 0:
        raise ValueError("tool length must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(53)
    truth_q = np.array([.35, -1., 1.4, -.6, .9, .2])
    true_flange = fk(truth_q)[0]
    tfg = make_t(np.eye(3), [0, 0, args.tool_length_mm/1000])
    desired_tcp = true_flange@tfg
    # These define a synthetic camera/object scene consistent with a reachable TCP.
    tbc = make_t(dh(.4, 0, 0, .3)[:3, :3], [.15, -.08, .35])
    tog = make_t(dh(-.3, 0, 0, .5)[:3, :3], [.025, -.015, .045])
    tco = np.linalg.inv(tbc)@desired_tcp@np.linalg.inv(tog)
    reconstructed_tcp = tbc@tco@tog
    correct_flange = reconstructed_tcp@np.linalg.inv(tfg)
    initial = truth_q+rng.normal(0, .18, 6)
    # Check geometric Jacobian with a finite difference independent of the IK solve.
    _, jac, _ = fk(initial)
    fd = np.zeros((6, 6)); eps = 1e-6
    for j in range(6):
        dq = np.eye(6)[j]*eps
        plus, minus = fk(initial+dq)[0], fk(initial-dq)[0]
        fd[:, j] = np.r_[(plus[:3, 3]-minus[:3, 3])/(2*eps), log_rotation(plus[:3, :3]@minus[:3, :3].T)/(2*eps)]
    jac_error = float(np.max(np.abs(fd-jac)))
    assert jac_error < 1e-6
    rows, histories, solutions = {}, {}, {}
    for name, target in [("correct_tool_transform", correct_flange), ("tcp_mistaken_for_flange", reconstructed_tcp)]:
        started = time.perf_counter()
        q, history, status = ik(target, initial)
        elapsed = (time.perf_counter()-started)*1000
        estimated_flange = fk(q)[0]
        tcp_error = pose_error(desired_tcp, estimated_flange@tfg)
        solver_error = pose_error(target, estimated_flange)
        rows[name] = dict(status=status, iterations=len(history), q_rad=q.tolist(),
            solver_position_mm=float(np.linalg.norm(solver_error[:3])*1000), solver_angle_deg=float(np.rad2deg(np.linalg.norm(solver_error[3:]))),
            actual_tcp_error_mm=float(np.linalg.norm(tcp_error[:3])*1000), actual_tcp_angle_deg=float(np.rad2deg(np.linalg.norm(tcp_error[3:]))), solve_ms=elapsed,
            within_joint_limits=bool(np.all(np.abs(q)<=LIMIT)))
        histories[name], solutions[name] = history, q
    unreachable = correct_flange.copy(); unreachable[:3, 3] = [2., 2., 2.]
    uq, _, ustatus = ik(unreachable, initial)
    report = dict(seed=53, environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        model="synthetic standard-DH 6R arm, not a hardware model", frame="T_AB maps B to A; B base, C camera, O object, G TCP/grasp, F flange", units="m/rad; reports mm/deg", timestamp="all transforms synthetic t=0", tool_length_mm=args.tool_length_mm,
        dh=dict(a_m=A.tolist(), d_m=D.tolist(), alpha_rad=ALPHA.tolist()), joint_names=[f"joint_{i}" for i in range(1, 7)], joint_limits_rad=LIMIT.tolist(), initial_q_rad=initial.tolist(),
        jacobian_max_abs_error=jac_error, chain_max_abs_error=float(np.max(np.abs(correct_flange-true_flange))), cases=rows,
        unreachable=dict(status=ustatus, residual_m=float(np.linalg.norm(pose_error(unreachable,fk(uq)[0])[:3]))),
        success_rule="flange translation <0.01 mm and orientation <0.0001 rad; kinematic only", collision_checked=False, robot_interactions=0, training_steps=0)
    (args.out_dir/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(args.out_dir/"interface_packet.npz", T_BC=tbc, T_CO=tco, T_OG=tog, T_FG=tfg, T_BG=desired_tcp, T_BF=correct_flange, q_seed=initial, q_solution=solutions["correct_tool_transform"], timestamp_s=0.)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, h in histories.items():
        axes[0].semilogy(np.maximum(h[:, 0], 1e-9), label=name)
        axes[1].semilogy(np.maximum(h[:, 1], 1e-9), label=name)
    for ax, label in zip(axes, ["flange task error (mm)", "orientation task error (deg)"]):
        ax.set(xlabel="IK iteration", ylabel=label); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle("Both IK tasks can converge; only the correct frame contract reaches the intended TCP")
    fig.savefig(args.out_dir/"ik-convergence.png", dpi=150); plt.close(fig)
    fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, wspace=.22)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    for name, q in solutions.items():
        tf, _, p = fk(q)
        pg = (tf@tfg)[:3, 3]
        ax.plot(*p.T, "o-", label=name, markersize=3)
        ax.plot(*np.vstack([tf[:3, 3], pg]).T, "--", color="black")
        ax.scatter(*pg, s=25)
    ax.scatter(*desired_tcp[:3, 3], marker="*", s=100, color="red", label="intended TCP")
    ax.set(xlabel="base x (m)", ylabel="base y (m)", zlabel="base z (m)", title="6R links and rigid tool offset")
    ax.legend(fontsize=6)
    ax.zaxis.labelpad = 1
    bx = fig.add_subplot(grid[0, 1])
    bx.bar(["correct", "TCP as flange"], [rows[k]["actual_tcp_error_mm"] for k in rows])
    bx.set(ylabel="actual TCP position error (mm)", title="Evaluate the physical point,\nnot only solver residual")
    fig.savefig(args.out_dir/"tool-frame-error.png", dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
