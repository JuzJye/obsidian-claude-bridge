"""Day49: a paired 2x2 ablation, scene-cluster bootstrap and domain stress test."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["mean", "robust", "plane", "robust+plane"]


def estimate(points, robust, plane):
    t = points.mean(axis=0)
    if robust:
        for _ in range(12):
            residual = np.linalg.norm(points-t, axis=1)
            w = np.minimum(1, 0.005 / np.maximum(residual, 1e-12))
            t = (w[:, None]*points).sum(axis=0)/w.sum()
    if plane:
        t[2] = 0.50  # an assumed known object-centre plane, not a learned fact
    return t


def paired_ci(differences, rng, repeats=2000):
    # Rows are independent scenes. Keep the four views within a scene together.
    scene_means = differences.mean(axis=1)
    n = len(scene_means)
    indices = rng.integers(0, n, size=(repeats, n))
    boot = scene_means[indices].mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=49)
    ap.add_argument("--outlier-rate", type=float, default=0.25)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day49/results.json")
    args = ap.parse_args()
    assert 0 <= args.outlier_rate < 1
    rng = np.random.default_rng(args.seed)
    scenes, views, measurements = 30, 4, 24
    centers = rng.uniform([-.15, -.15, 0], [.15, .15, 0], (scenes, 3))
    scene_bias = rng.normal(0, .002, (scenes, 1, 1, 3))
    sensor_noise = rng.normal(0, .003, (scenes, views, measurements, 3))
    outlier = rng.random((scenes, views, measurements, 1)) < args.outlier_rate
    error = scene_bias + sensor_noise + outlier * [0.025, 0.005, 0.040]
    results, csv_rows, errors = {}, [], {}
    for domain, height in [("nominal", .50), ("raised", .52)]:
        truth = centers.copy(); truth[:, 2] = height
        observed = truth[:, None, None, :] + error
        e = np.empty((4, scenes, views)); latency = [[] for _ in METHODS]
        for m, name in enumerate(METHODS):
            for s in range(scenes):
                for v in range(views):
                    started = time.perf_counter_ns()
                    predicted = estimate(observed[s, v], m in [1, 3], m in [2, 3])
                    latency[m].append((time.perf_counter_ns()-started)/1e6)
                    e[m, s, v] = np.linalg.norm(predicted-truth[s])*1000
                    csv_rows.append([domain, name, s, v, e[m, s, v], int(e[m, s, v] <= 10)])
        errors[domain] = e
        reports = {}
        for m, name in enumerate(METHODS):
            reports[name] = dict(attempts=scenes*views, successes=int((e[m] <= 10).sum()),
                mean_mm=float(e[m].mean()), median_mm=float(np.median(e[m])),
                p90_mm=float(np.quantile(e[m], .9)),
                paired_delta_vs_mean_mm=float((e[m]-e[0]).mean()),
                scene_bootstrap_95_ci_mm=paired_ci(e[m]-e[0], rng),
                median_estimator_ms=float(np.median(latency[m])))
        # Negative means more than additive error reduction for these four cells.
        interaction = e[3]-e[1]-e[2]+e[0]
        results[domain] = dict(methods=reports,
            interaction_mm=float(interaction.mean()),
            interaction_ci_mm=paired_ci(interaction, rng))
    report = dict(python=platform.python_version(), numpy=np.__version__,
        matplotlib=matplotlib.__version__, device="CPU", seed=args.seed,
        outlier_rate=args.outlier_rate, scenes=scenes, views_per_scene=views,
        measurements_per_view=measurements, data="synthetic translation-only observations",
        modalities="3D measurements with known point association", unit="m internally, mm reported",
        frame="single camera frame", timestamps="static independent scenes, four correlated views",
        training_samples=0, optimizer_iterations=12, real_robot_interactions=0,
        success="translation error <= 10 mm; not physical grasp success", results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "method", "scene", "view", "error_mm", "success"])
        writer.writerows(csv_rows)
    print(json.dumps(report, indent=2))
    if args.output.stem != "results":
        return
    asset = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day49"
    asset.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, domain in zip(axes, ["nominal", "raised"]):
        e = errors[domain]
        for s in range(scenes):
            ax.plot(range(4), e[:, s].mean(axis=1), color="gray", alpha=.18)
        ax.plot(range(4), e.mean(axis=(1, 2)), "o-", lw=2, label="mean of 30 scenes")
        ax.axhline(10, color="tomato", ls="--", label="10 mm proxy threshold")
        ax.set(xticks=range(4), xticklabels=METHODS, title=domain, ylabel="translation error (mm)")
        ax.tick_params(axis="x", rotation=15); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(asset/"paired_ablation.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for shift, domain in [(-.12, "nominal"), (.12, "raised")]:
        reports = results[domain]["methods"]
        values = np.array([reports[n]["paired_delta_vs_mean_mm"] for n in METHODS[1:]])
        intervals = np.array([reports[n]["scene_bootstrap_95_ci_mm"] for n in METHODS[1:]])
        ax.errorbar(values, np.arange(3)+shift,
            xerr=np.maximum(0, np.vstack([values-intervals[:, 0], intervals[:, 1]-values])),
            fmt="o", capsize=4, label=domain)
    ax.axvline(0, color="black", lw=1); ax.set(yticks=range(3), yticklabels=METHODS[1:],
        xlabel="paired mean error difference versus mean (mm); negative is better")
    ax.grid(alpha=.2); ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(asset/"grouped_confidence.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
