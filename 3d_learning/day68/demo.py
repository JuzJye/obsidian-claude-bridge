"""D68: CPU multimodal one-step dynamics; synthetic feature observations only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

DT, MASS, DAMPING = .02, .5, 1.


def episodes(rng, count, stiffness, bias=0.):
    """Independent 30-step episodes; vision proxy [x,v], touch proxy [force,contact]."""
    rows, targets, ids = [], [], []
    for e in range(count):
        k = rng.uniform(*stiffness)
        x, v = rng.uniform(-.02, .025), rng.uniform(-.08, .08)
        commands = rng.uniform(-1., 1., 6)
        for step in range(30):
            u = commands[step // 5]
            force = k * max(x, 0.)
            ox, ov = x + rng.normal(0, .0003), v + rng.normal(0, .005)
            measured_force = force + rng.normal(0, .015) + bias
            obs = [ox, ov, measured_force, float(measured_force > .02), u]
            vn = v + DT / MASS * (u - DAMPING * v - force)
            xn = x + DT * vn  # semi-implicit Euler; simplified unilateral spring
            rows.append(obs); targets.append([xn, vn]); ids.append(e)
            x, v = xn, vn
    return np.asarray(rows, np.float32), np.asarray(targets, np.float32), np.asarray(ids)


class Dynamics(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.vision = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.touch = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.action = nn.Sequential(nn.Linear(1, 8), nn.Tanh())
        self.transition = nn.Sequential(nn.Linear(40, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        vis = self.vision(x[:, :2])
        tac = self.touch(x[:, 2:4])
        act = self.action(x[:, 4:5])
        if self.mode == "vision_action":
            tac = torch.zeros_like(tac)
        if self.mode == "vision_touch":
            act = torch.zeros_like(act)
        return self.transition(torch.cat([vis, tac, act], dim=-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-touch-bias", type=float, default=0.)
    ap.add_argument("--out", type=Path, default=Path("_assets/3d_learning/day68/default"))
    args = ap.parse_args()
    if not np.isfinite(args.test_touch_bias):
        ap.error("bias must be finite")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train = episodes(np.random.default_rng(680), 96, (20, 80))
    val = episodes(np.random.default_rng(681), 32, (20, 80))
    test = episodes(np.random.default_rng(682), 32, (20, 80), args.test_touch_bias)
    ood = episodes(np.random.default_rng(683), 32, (100, 150), args.test_touch_bias)
    xmean, xstd = train[0].mean(0), np.maximum(train[0].std(0), 1e-6)
    residual = train[1] - train[0][:, :2]
    ymean, ystd = residual.mean(0), np.maximum(residual.std(0), 1e-6)
    tx = torch.from_numpy((train[0] - xmean) / xstd)
    ty = torch.from_numpy((residual - ymean) / ystd)
    vx = torch.from_numpy((val[0] - xmean) / xstd)
    vy = torch.from_numpy((val[1] - val[0][:, :2] - ymean) / ystd)

    def predict(net, x):
        with torch.no_grad():
            delta = net(torch.from_numpy((x - xmean) / xstd)).numpy() * ystd + ymean
        return x[:, :2] + delta

    def score(pred, truth):
        error = pred - truth
        contact = truth[:, 0] > 0
        correct = (pred[:, 0] > 0) == contact
        return {"count": len(truth), "x_rmse_mm": float(np.sqrt(np.mean(error[:, 0]**2)) * 1000),
                "v_rmse_mm_s": float(np.sqrt(np.mean(error[:, 1]**2)) * 1000),
                "contact_correct": int(correct.sum()), "contact_count": int(contact.sum()),
                "contact_agreement": float(correct.mean()),
                "contact_v_rmse_mm_s": float(np.sqrt(np.mean(error[contact, 1]**2)) * 1000),
                "free_v_rmse_mm_s": float(np.sqrt(np.mean(error[~contact, 1]**2)) * 1000)}

    models, report, histories, predictions = {}, {}, {}, {}
    tic = time.perf_counter()
    for mode in ["vision_action", "vision_touch", "vision_touch_action"]:
        torch.manual_seed(68)
        net = Dynamics(mode)
        opt = torch.optim.Adam(net.parameters(), lr=.008)
        history, best, best_state, best_epoch = [], float("inf"), None, None
        for epoch in range(300):
            net.train(); opt.zero_grad()
            loss = ((net(tx) - ty)**2).mean()
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(((net(vx) - vy)**2).mean())
            history.append([float(loss.detach()), vl])
            if vl < best:
                best, best_epoch = vl, epoch + 1
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state); net.eval(); models[mode] = net
        histories[mode] = np.array(history)
        report[mode] = {"best_epoch": best_epoch, "validation_normalized_mse": best,
                        "total_parameters": sum(p.numel() for p in net.parameters()),
                        "weights_sha256": hashlib.sha256(b"".join(v.numpy().tobytes() for v in best_state.values())).hexdigest()}
        for name, data in [("test", test), ("ood_stiffness", ood)]:
            pred = predict(net, data[0])
            report[mode][name] = score(pred, data[1])
            predictions[mode + "_" + name] = pred
    training_seconds = time.perf_counter() - tic
    # Wall-clock CPU batch inference, including normalization and NumPy conversion.
    for mode, net in models.items():
        latency = {}
        for batch_size in (1, 960):
            sample = test[0][:batch_size]
            for _ in range(10):
                predict(net, sample)
            timings = []
            for _ in range(100):
                start = time.perf_counter()
                predict(net, sample)
                timings.append((time.perf_counter() - start) * 1000)
            latency[str(batch_size)] = {"median_ms": float(np.median(timings)),
                                        "p95_ms": float(np.quantile(timings, .95))}
        report[mode]["cpu_prediction_latency_by_batch_size"] = latency
    report["persistence"] = {name: score(data[0][:, :2], data[1])
                             for name, data in [("test", test), ("ood_stiffness", ood)]}
    actions = np.linspace(-1, 1, 81, dtype=np.float32)
    sweep = np.tile([.015, 0, 70 * .015 + args.test_touch_bias, 1, 0], (81, 1)).astype(np.float32)
    sweep[:, 4] = actions
    truth_v = DT / MASS * (actions - 70 * .015)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    colors = ["#b36c24", "#9d426b", "#217f91"]
    for (mode, net), color in zip(models.items(), colors):
        axes[0].plot(histories[mode][:, 1], label=mode, color=color)
        axes[1].plot(actions, predict(net, sweep)[:, 1] * 1000, label=mode, color=color)
    axes[0].set(xlabel="Optimizer step", ylabel="Validation normalized MSE", yscale="log")
    axes[0].legend(fontsize=7)
    axes[1].plot(actions, truth_v * 1000, "k--", label="analytic next velocity")
    axes[1].set(xlabel="Candidate force command [N]", ylabel="Next velocity [mm/s]",
                title="Same observation; only action changes")
    axes[1].legend(fontsize=7)
    labels = list(models) + ["persistence"]
    ix = np.arange(4)
    for shift, split, color in [(-.18, "test", "#217f91"), (.18, "ood_stiffness", "#b36c24")]:
        axes[2].bar(ix + shift, [report[m][split]["v_rmse_mm_s"] for m in labels],
                    width=.36, label=split, color=color)
    axes[2].set(xticks=ix, xticklabels=["V+A", "V+T", "V+T+A", "persist"],
                ylabel="Velocity RMSE [mm/s]", title="All 960 transitions per split")
    axes[2].legend(fontsize=8)
    fig.savefig(args.out / "action_and_dynamics.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for mode, color in zip(models, colors):
        pred = predictions[mode + "_test"]
        axes[0].scatter(test[1][::8, 1] * 1000, pred[::8, 1] * 1000, s=9, alpha=.5, label=mode, color=color)
    lo, hi = test[1][:, 1].min() * 1000, test[1][:, 1].max() * 1000
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set(xlabel="True next velocity [mm/s]", ylabel="Predicted next velocity [mm/s]")
    axes[0].legend(fontsize=7)
    for shift, key, color in [(-.18, "free_v_rmse_mm_s", "#b36c24"), (.18, "contact_v_rmse_mm_s", "#217f91")]:
        axes[1].bar(ix + shift, [report[m]["test"][key] for m in labels], .36, label=key, color=color)
    axes[1].set(xticks=ix, xticklabels=["V+A", "V+T", "V+T+A", "persist"],
                ylabel="Velocity RMSE [mm/s]", title="Regimes defined by true next position")
    axes[1].legend(fontsize=7)
    fig.savefig(args.out / "regimes_and_residuals.png", dpi=160); plt.close(fig)
    np.savez_compressed(args.out / "predictions.npz", **predictions, test_x=test[0], test_y=test[1],
                        test_episode_id=test[2], train_x=train[0], train_y=train[1], train_episode_id=train[2],
                        xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd)
    torch.save({mode: net.state_dict() for mode, net in models.items()}, args.out / "weights.pt")
    result = {"source": "synthetic_1d_unilateral_spring_feature_observations", "dt_s": DT, "mass_kg": MASS,
              "damping_Ns_m": DAMPING, "test_touch_bias_N": args.test_touch_bias,
              "created_utc": datetime.now(timezone.utc).isoformat(), "seed_train_val_test_ood": [680, 681, 682, 683],
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "matplotlib": matplotlib.__version__, "device": "cpu"},
              "split_episodes": {"train": 96, "validation": 32, "test": 32, "ood": 32},
              "steps_per_episode": 30, "optimizer_steps_each": 300, "training_seconds": training_seconds,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "robot_attempts": 0, "metrics": report,
              "limits": "One-step only; 1D spring; no rendered RGB, physical tactile sensor, rotation or closed-loop task success."}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

