#!/usr/bin/env python3
"""D70: separate fusion/dynamics ledgers; paired episode bootstrap on CPU.
Synthetic analytic spring + fixed estimators; no model training or robot trials.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT, MASS, DAMP, N, FRAMES, H = .02, .5, 1., 48, 8, 10
METHODS = ("baseline", "fusion")
CONFIG = dict(dt_s=DT, mass_kg=MASS, damping_Ns_m=DAMP, stiffness_prior_N_m=50.,
              depth_weight=.92, ridge_m2=1e-5, contact_threshold_N=.05,
              probability_temperature_N=.06, stiffness_clip_N_m=[5., 200.])


def digest(a):
    return hashlib.sha256(a if isinstance(a, bytes) else a.tobytes()).hexdigest()


def step(s, u, k):
    v = s[..., 1] + DT / MASS * (u - DAMP * s[..., 1] - k * np.maximum(s[..., 0], 0))
    return np.stack([s[..., 0] + DT * v, v], axis=-1)


def make_data(seed):
    r = np.random.default_rng(seed)
    k = r.uniform(20., 80., N)
    state = np.zeros((N, FRAMES + H, 2))
    state[:, 0] = np.stack([r.uniform(-.008, .018, N), r.uniform(-.05, .05, N)], -1)
    action = np.repeat(r.uniform(-.8, .8, (N, 6)), 3, axis=1)[:, :FRAMES + H - 1]
    for t in range(FRAMES + H - 1):
        state[:, t + 1] = step(state[:, t], action[:, t], k)
    p = np.zeros((N, FRAMES, 3))
    p[:, :, 0] = .45 + state[:, :FRAMES, 0]
    p[:, :, 1] = r.uniform(-.04, .04, (N, 1))
    p[:, :, 2] = r.uniform(.20, .30, (N, 1))
    force = k[:, None] * np.maximum(state[:, :FRAMES, 0], 0)
    scale = np.where(np.arange(N) < 36, 1., 5.)[:, None, None]
    # Shared episode bias makes the eight frame errors correlated.
    visual = p + scale * (.0012 * r.normal(size=(N, 1, 3)) + .0008 * r.normal(size=p.shape))
    depth = p + scale * (.00035 * r.normal(size=(N, 1, 3)) + .00025 * r.normal(size=p.shape))
    velocity = state[:, :FRAMES, 1] + .004 * r.normal(size=(N, 1)) + .002 * r.normal(size=(N, FRAMES))
    tactile = force + .01 * r.normal(size=(N, 1)) + .01 * r.normal(size=force.shape)
    return dict(state=state, position=p, force=force, contact=(force > .05).astype(float),
                visual=visual, depth=depth, velocity=velocity, tactile=tactile,
                action=action[:, FRAMES - 1:], stiffness=k)


def predict(obs, future_action, fusion):
    # Only historical observations and proposed actions enter prediction.
    p = .08 * obs["visual"] + .92 * obs["depth"] if fusion else obs["visual"].copy()
    penetration = np.maximum(p[:, :, 0] - .45, 0.)
    k = np.full(N, 50.)
    if fusion:
        k = np.clip((np.sum(penetration * obs["tactile"], axis=1) + 1e-5 * 50.) /
                    (np.sum(penetration ** 2, axis=1) + 1e-5), 5., 200.)
    f = obs["tactile"] if fusion else 50. * penetration
    prob = 1. / (1. + np.exp(-np.clip((f - .05) / .06, -40, 40)))
    s = np.stack([p[:, -1, 0] - .45, obs["velocity"][:, -1]], -1)
    trajectory = np.zeros((N, H, 2))
    for t in range(H):
        s = step(s, future_action[:, t], k)
        trajectory[:, t] = s
    return dict(position=p, probability=prob, trajectory=trajectory, stiffness=k)


def episode_losses(data, pred):
    e = pred["trajectory"] - data["state"][:, FRAMES:]
    losses = {"position_mm": np.mean(np.sum((pred["position"] - data["position"]) ** 2, -1), -1),
              "contact_brier": np.mean((pred["probability"] - data["contact"]) ** 2, -1)}
    for h in (1, 5, 10):
        losses[f"x{h}_mm"] = e[:, h - 1, 0] ** 2
        losses[f"v{h}_mm_s"] = e[:, h - 1, 1] ** 2
    losses["x_path_mm"] = np.mean(e[:, :, 0] ** 2, -1)
    losses["v_path_mm_s"] = np.mean(e[:, :, 1] ** 2, -1)
    return losses


def aggregate(loss, key, axis=None):
    value = np.mean(loss, axis=axis)
    return value if key == "contact_brier" else 1000. * np.sqrt(value)


def audit(expected, returned):
    missing, extra = sorted(set(expected) - set(returned)), sorted(set(returned) - set(expected))
    duplicate = len(returned) != len(set(returned))
    return dict(valid=not (missing or extra or duplicate), expected=len(expected), returned=len(returned),
                missing_ids=missing, extra_ids=extra, duplicate_ids=duplicate)


def paired_report(losses, keep, rng, reps, valid):
    indices = rng.integers(0, N, (reps, N))
    result = {}
    for key in losses["baseline"]:
        a, b = losses["baseline"][key], losses["fusion"][key]
        naive = {m: float(aggregate(losses[m][key][keep[m]], key)) for m in METHODS}
        delta = aggregate(b[indices], key, 1) - aggregate(a[indices], key, 1)
        result[key] = dict(baseline=naive["baseline"], fusion=naive["fusion"] if valid else None,
                           delta_fusion_minus_baseline=float(aggregate(b, key) - aggregate(a, key)) if valid else None,
                           paired_episode_percentile95=np.quantile(delta, [.025, .975]).tolist() if valid else None,
                           naive_available_case=naive)
    return result


def figures(out, losses, metrics, keep, valid):
    colors = ["#496f9c", "#df7d38"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for j, key in enumerate(("position_mm", "contact_brier")):
        for m, color in zip(METHODS, colors):
            vals = np.sqrt(losses[m][key]) * 1000 if j == 0 else losses[m][key]
            axes[j].plot(keep[m], vals[keep[m]], ".", color=color, label=m)
        axes[j].set(xlabel="Episode ID (same eight-frame cluster)", ylabel="Translation RMSE (mm)" if j == 0 else "Contact Brier",
                    title="Available cases; never a full-score replacement")
        axes[j].axvline(35.5, color="gray", linestyle=":")
    axes[0].legend()
    axes[2].bar(METHODS, [len(keep[m]) for m in METHODS], color=colors)
    axes[2].axhline(N, color="black", linestyle="--", label="Manifest = 48")
    axes[2].set(ylim=(0, N + 6), ylabel="Returned episodes", title="Integrity PASS" if valid else "Integrity INVALID")
    axes[2].legend()
    fig.suptitle("Current perception: 3D translation proxy and contact, NOT full 6D pose")
    fig.savefig(out / "fusion.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for j, prefix in enumerate(("x", "v")):
        for m, color in zip(METHODS, colors):
            vals = [metrics[f"{prefix}{h}_" + ("mm" if j == 0 else "mm_s")]["naive_available_case"][m] for h in (1, 5, 10)]
            axes[j].plot([1, 5, 10], vals, "o-", color=color, label=m)
        axes[j].set(xlabel="Prediction horizon (dt = 0.02 s)", ylabel="RMSE (mm)" if j == 0 else "RMSE (mm/s)",
                    title="Free rollout" if valid else "Available-case only: invalid comparison")
    axes[0].legend()
    keys = ["position_mm", "x1_mm", "x5_mm", "x10_mm"]
    if valid:
        for i, key in enumerate(keys):
            lo, hi = metrics[key]["paired_episode_percentile95"]
            axes[2].plot([lo, hi], [i, i], "-", color=colors[1], linewidth=3)
            axes[2].plot(metrics[key]["delta_fusion_minus_baseline"], i, "o", color="black")
        axes[2].axvline(0, color="gray", linestyle="--")
        axes[2].set(yticks=range(4), yticklabels=["current position", "future x @1", "future x @5", "future x @10"],
                    xlabel="Fusion - baseline (mm); negative is better", title="Paired episode bootstrap 95% CI")
    else:
        axes[2].axis("off")
        axes[2].text(.04, .62, "Official fusion scores = null\nPaired differences / CI = null\nMissing episodes cannot be inner-joined away.", transform=axes[2].transAxes)
    fig.suptitle("Dynamics is reported separately; no blended overall score")
    fig.savefig(out / "dynamics.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--drop-hard", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "_assets/3d_learning/day70/default")
    args = ap.parse_args()
    if not 0 <= args.drop_hard < N or args.bootstrap < 1000:
        ap.error("Require 0 <= drop-hard < 48 and bootstrap >= 1000")
    args.out.mkdir(parents=True, exist_ok=True)
    data, predictions, latency = make_data(args.seed), {}, {}
    obs = {k: data[k] for k in ("visual", "depth", "velocity", "tactile")}
    for m in METHODS:
        start = time.perf_counter()
        predictions[m] = predict(obs, data["action"], m == "fusion")
        latency[m] = 1000 * (time.perf_counter() - start)
    losses = {m: episode_losses(data, predictions[m]) for m in METHODS}
    keep = {m: np.arange(N) for m in METHODS}
    if args.drop_hard:
        removed = np.argsort(losses["fusion"]["position_mm"])[-args.drop_hard:]
        keep["fusion"] = np.setdiff1d(keep["fusion"], removed)
    expected = [f"ep{i:03d}" for i in range(N)]
    checks = {m: audit(expected, [expected[i] for i in keep[m]]) for m in METHODS}
    valid = all(v["valid"] for v in checks.values())
    metrics = paired_report(losses, keep, np.random.default_rng(args.seed + 1), args.bootstrap, valid)
    # Explicitly reject ground-truth dependencies in the official predictor API.
    assert set(obs) == {"visual", "depth", "velocity", "tactile"}
    assert all(np.all(np.isfinite(p["trajectory"])) for p in predictions.values())
    manifest = dict(expected_episode_ids=expected, frames_per_episode=FRAMES, future_steps=H,
                    history_t_s=(np.arange(FRAMES) * DT - (FRAMES - 1) * DT).tolist(),
                    future_t_s=(np.arange(1, H + 1) * DT).tolist(), coordinate="fixed synthetic world, +x penetration",
                    observation_units=dict(position="m", velocity="m/s", tactile="N"), action="force N, zero-order hold dt",
                    split="independent synthetic evaluation episodes; no training/validation fit", config=CONFIG,
                    loss_storage_units={k: "dimensionless" if k == "contact_brier" else "(m/s)^2" if k.startswith("v") else "m^2" for k in metrics},
                    contact_probability="fixed sigmoid score, not independently calibrated",
                    resampling="paired episodes from empirical 36 normal / 12 noisy mixture; not individual frames")
    report = dict(day=70, seed=args.seed, bootstrap_seed=args.seed + 1, bootstrap_reps=args.bootstrap,
                  evidence="synthetic analytic spring and fixed estimators; not a trained multimodal model",
                  environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
                  script_sha256=digest(Path(__file__).read_bytes()), data_sha256=digest(b"".join(data[k].tobytes() for k in sorted(data))),
                  parameter_sha256=digest(json.dumps(CONFIG, sort_keys=True).encode()), drop_hard=args.drop_hard,
                  integrity=dict(status="pass" if valid else "invalid", methods=checks),
                  fusion={k: metrics[k] for k in ("position_mm", "contact_brier")},
                  dynamics={k: v for k, v in metrics.items() if k not in ("position_mm", "contact_brier")},
                  latency_ms=latency, latency_scope="one CPU batch of 48 histories and ten-step predictions; excludes data, evaluation, bootstrap and plotting; not sensor-to-action latency",
                  trial_counts=dict(synthetic_episodes=N, perception_frames=N*FRAMES, history_intervals=N*(FRAMES-1), future_intervals=N*H, robot_attempts=0, physical_simulator_trials=0, training_updates=0),
                  acceptance=dict(material_integrity="pass" if valid else "invalid", project="pending_real_data_and_task_evidence", personal="not_assessed"))
    for name, value in (("manifest.json", manifest), ("report.json", report)):
        (args.out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for m in METHODS:
        with (args.out / f"{m}_ledger.jsonl").open("w") as handle:
            for i in keep[m]:
                row = dict(episode_id=expected[i], losses={k: float(v[i]) for k, v in losses[m].items()},
                           predictions={k: np.asarray(v[i]).tolist() for k, v in predictions[m].items()})
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez(args.out / "truth.npz", **data)
    figures(args.out, losses, metrics, keep, valid)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
