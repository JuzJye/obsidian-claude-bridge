"""D54: quintic timing and timestamp-aware replay; CPU kinematic teaching model."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q0 = np.array([0., -.5, .6, -.2, .3, 0.])
Q1 = np.array([.35, -1., 1.4, -.6, .9, .2])
VMAX = np.array([.7, .6, .8, 1., 1., 1.2])
AMAX = np.array([1.2, 1., 1.4, 2., 2., 2.])
DT = .02
JOINTS = [f"joint_{i}" for i in range(1, 7)]


def profile(t, duration):
    tau = np.asarray(t)/duration
    s = 10*tau**3-15*tau**4+6*tau**5
    ds = (30*tau**2-60*tau**3+30*tau**4)/duration
    dds = (60*tau-180*tau**2+120*tau**3)/duration**2
    return Q0+s[:, None]*(Q1-Q0), ds[:, None]*(Q1-Q0), dds[:, None]*(Q1-Q0)


def replay(records, actual_time=True):
    q = Q0.copy(); states = [q.copy()]; expected_start = 0.
    for k, row in enumerate(records):
        if row["sequence"] != k or row["joint_names"] != JOINTS:
            raise ValueError("sequence gap/duplicate or joint order mismatch")
        if row["action_type"] != "joint_velocity" or row["unit"] != "rad/s":
            raise ValueError("unsupported action contract")
        if not np.isclose(row["apply_start_s"], expected_start, atol=1e-12, rtol=0):
            raise ValueError("applied intervals are not contiguous")
        applied_dt = row["apply_end_s"]-row["apply_start_s"]
        if applied_dt <= 0 or not np.isclose(applied_dt, row["dt_applied_s"], atol=1e-12, rtol=0):
            raise ValueError("invalid applied duration")
        dt = applied_dt if actual_time else row["dt_nominal_s"]
        velocity = np.asarray(row["command"], dtype=float)
        if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
            raise ValueError("invalid command")
        q += velocity*dt
        states.append(q.copy()); expected_start = row["apply_end_s"]
    return np.array(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stall-periods", type=float, default=4.)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day54")
    args = parser.parse_args()
    if args.stall_periods < 0:
        raise ValueError("stall periods must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    distance = np.abs(Q1-Q0)
    velocity_bound = 1.875*distance/VMAX
    acceleration_bound = np.sqrt((10*np.sqrt(3)/3)*distance/AMAX)
    tmin = float(max(velocity_bound.max(), acceleration_bound.max()))
    duration = float(np.ceil(tmin/DT)*DT)
    n = int(round(duration/DT))
    planned_t = np.arange(n+1)*DT
    q, dq, ddq = profile(planned_t, duration)
    analytic_vratio = float(np.max(1.875*distance/duration/VMAX))
    analytic_aratio = float(np.max((10*np.sqrt(3)/3)*distance/duration**2/AMAX))
    assert analytic_vratio <= 1+1e-12 and analytic_aratio <= 1+1e-12
    assert np.allclose(q[0], Q0) and np.allclose(q[-1], Q1)
    assert np.allclose(dq[[0, -1]], 0) and np.allclose(ddq[[0, -1]], 0)
    command = np.diff(q, axis=0)/DT  # exact interval-average velocity of the sampled path
    durations = DT*(1+.25*np.sin(2*np.pi*(np.arange(n)+.5)/n))
    durations[n//2] += args.stall_periods*DT  # delayed next replacement: hold the active command longer
    actual_t = np.r_[0., np.cumsum(durations)]
    measured = np.vstack([Q0, Q0+np.cumsum(command*durations[:, None], axis=0)])
    rows = []
    for k in range(n):
        rows.append(dict(sequence=k, chunk_id=k//10, chunk_index=k%10,
            nominal_horizon=10, executed_horizon=min(10, n-(k//10)*10), planned_start_s=float(planned_t[k]),
            apply_start_s=float(actual_t[k]), apply_end_s=float(actual_t[k+1]),
            dt_applied_s=float(durations[k]), dt_nominal_s=DT,
            clock="single synthetic monotonic clock", observation_time_s=None, policy_inference_ms=None, joint_names=JOINTS,
            action_type="joint_velocity", unit="rad/s", command=command[k].tolist(),
            measured_before_rad=measured[k].tolist(), measured_after_rad=measured[k+1].tolist(),
            feedback_source="ideal kinematic integrator; not encoder feedback", ack="applied_in_simulation"))
    logfile = args.out_dir/"action-log.jsonl"
    logfile.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    loaded = [json.loads(line) for line in logfile.read_text().splitlines()]
    replayed, nominal = replay(loaded), replay(loaded, actual_time=False)
    exact_error = float(np.max(np.abs(replayed-measured)))
    assert exact_error < 1e-12
    duplicate = loaded[:5]+[loaded[4]]+loaded[5:]
    try:
        replay(duplicate)
    except ValueError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    assert duplicate_rejected
    # An action-unit bug: treat each rad/s value as a rad displacement.
    wrong_increment = Q0+command.sum(axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(planned_t, q); axes[0].set(ylabel="planned q (rad)", title="One geometric path, a feasible quintic time law")
    axes[1].plot(planned_t, np.abs(dq)/VMAX); axes[1].axhline(1, color="red", ls="--")
    axes[1].set(ylabel="|velocity| / limit")
    axes[2].plot(planned_t, np.abs(ddq)/AMAX); axes[2].axhline(1, color="red", ls="--")
    axes[2].set(ylabel="|acceleration| / limit", xlabel="planned time (s)")
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(JOINTS, ncol=3, fontsize=8)
    fig.savefig(args.out_dir/"time-parameterization.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(actual_t, measured[:, 2], label="simulated measured joint 3")
    axes[0].plot(actual_t, replayed[:, 2], "--", label="replay: applied duration")
    axes[0].plot(actual_t, nominal[:, 2], ":", label="replay: nominal duration")
    axes[0].set(xlabel="actual monotonic time (s)", ylabel="joint 3 (rad)", title="A timestamp is part of the action contract")
    axes[0].legend(fontsize=8)
    axes[1].plot(actual_t, np.linalg.norm(nominal-measured, axis=1), label="nominal replay discrepancy")
    axes[1].plot(actual_t, np.linalg.norm(replayed-measured, axis=1), "--", label="applied-time replay discrepancy")
    axes[1].set(xlabel="actual monotonic time (s)", ylabel="joint-vector error norm (rad)", title="Correct replay reconstructs drift; it does not remove it")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(args.out_dir/"log-replay.png", dpi=150); plt.close(fig)
    report = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        seed="deterministic; no random sampling", model="ideal joint-velocity integrator; no rigid-body dynamics or contact",
        joint_names=JOINTS, q0_rad=Q0.tolist(), q1_rad=Q1.tolist(), velocity_limits_rad_s=VMAX.tolist(), acceleration_limits_rad_s2=AMAX.tolist(),
        nominal_dt_s=DT, minimum_duration_s=tmin, rounded_duration_s=duration, intervals=n,
        time_scalings={"feasible":dict(duration_s=duration, maximum_velocity_ratio=analytic_vratio, maximum_acceleration_ratio=analytic_aratio),
            "half_duration":dict(duration_s=duration/2, maximum_velocity_ratio=2*analytic_vratio, maximum_acceleration_ratio=4*analytic_aratio)},
        stall_periods=args.stall_periods, actual_duration_s=float(actual_t[-1]), largest_applied_dt_s=float(durations.max()),
        replay_max_abs_rad=exact_error, nominal_replay_terminal_error_rad=float(np.linalg.norm(nominal[-1]-measured[-1])),
        nominal_replay_max_error_rad=float(np.max(np.linalg.norm(nominal-measured, axis=1))), simulated_terminal_goal_error_rad=float(np.linalg.norm(measured[-1]-Q1)),
        velocity_as_increment_terminal_goal_error_rad=float(np.linalg.norm(wrong_increment-Q1)), duplicate_rejected=duplicate_rejected,
        chunk_contract="10 rows per chunk except final partial chunk; velocity rad/s; nominal 50 Hz; full available chunk executed; no learned VLA invoked",
        collision_checked=False, torque_checked=False, actual_hardware_feedback=False, training_steps=0, robot_interactions=0,
        synthetic_applied_intervals=n, elapsed_ms=(time.perf_counter()-start)*1000)
    (args.out_dir/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(args.out_dir/"trajectory.npz", q=q, dq=dq, ddq=ddq, planned_time_s=planned_t, actual_time_s=actual_t, simulated_measured=measured, replayed=replayed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
