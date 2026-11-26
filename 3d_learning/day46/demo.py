"""D46: predeclared five-method paired ablation, independent synthetic scenes."""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["L2", "Huber", "Cauchy", "Precision", "Huber+Precision"]


def rigid(x, y, w):
    w = w / w.sum()
    xc, yc = w @ x, w @ y
    u, _, vt = np.linalg.svd((x-xc).T @ ((y-yc)*w[:, None]))
    r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    return r, yc-r@xc


def solve(x, y, sigma, method, delta=.006):
    r, t = rigid(x, y, np.ones(len(x)))
    for _ in range(25):
        e = np.linalg.norm(x@r.T+t-y, axis=1)
        w = np.ones(len(x))
        if "Huber" in method:
            w *= np.minimum(1., delta / np.maximum(e, 1e-12))
        if method == "Cauchy":
            w *= 1. / (1.+(e/delta)**2)
        if "Precision" in method:
            w *= 1. / sigma**2
        r, t = rigid(x, y, w)
    return r, t


def scene(seed, j):
    rng = np.random.default_rng(seed*1000+j)
    x = rng.uniform(-1, 1, (160, 3))*[.06, .045, .035]
    angle = rng.uniform(-1, 1)
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    t = np.r_[rng.uniform(-.05, .05, 2), rng.uniform(.5, .9)]
    sigma = np.full(len(x), .002)
    bad = rng.permutation(len(x))[:32]
    sigma[bad] = .012
    y = x@r.T+t+rng.normal(size=x.shape)*sigma[:, None]
    y[bad] += [.05, -.03, .06]
    probe = rng.uniform(-1, 1, (256, 3))*[.06, .045, .035]
    return x, y, sigma, r, t, probe


def evaluate(seeds, count, flipped=False):
    rows = []
    digest = hashlib.sha256()
    for seed in seeds:
        for j in range(count):
            x, y, sigma, rt, tt, probe = scene(seed, j)
            digest.update(x.tobytes()+y.tobytes())
            supplied = np.where(sigma == .002, .012, .002) if flipped else sigma
            for method in METHODS:
                start = time.perf_counter()
                r, t = solve(x, y, supplied, method)
                latency = (time.perf_counter()-start)*1000
                trans = float(np.linalg.norm(t-tt)*1000)
                angle = float(np.rad2deg(np.arccos(np.clip((np.trace(r@rt.T)-1)/2, -1, 1))))
                add = float(np.linalg.norm(probe@r.T+t-(probe@rt.T+tt), axis=1).mean()*1000)
                rows.append(dict(seed=seed, scene_id=f"{seed}-{j}", method=method, add_mm=add, trans_mm=trans, angle_deg=angle, success=bool(trans<5 and angle<2), latency_ms=latency))
    summary = {}
    for method in METHODS:
        sel = [r for r in rows if r["method"] == method]
        seed_means = [float(np.mean([r["add_mm"] for r in sel if r["seed"]==s])) for s in seeds]
        summary[method] = dict(add_mean_mm=float(np.mean([r["add_mm"] for r in sel])), add_seed_std_mm=float(np.std(seed_means, ddof=1)), seed_means_mm=seed_means,
            successes=sum(r["success"] for r in sel), attempts=len(sel), median_solve_ms=float(np.median([r["latency_ms"] for r in sel])))
    return rows, summary, digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "_assets/3d_learning/day46")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_rows, val, val_hash = evaluate([461, 462, 463], 6)
    winner = min(METHODS, key=lambda k: val[k]["add_mean_mm"])
    test_rows, test, test_hash = evaluate([471, 472, 473], 12)
    ood_rows, ood, ood_hash = evaluate([471, 472, 473], 12, flipped=True)
    assert test_hash == ood_hash  # Only reported uncertainty changes.
    paired = {}
    baseline = {r["scene_id"]: r["add_mm"] for r in test_rows if r["method"]=="L2"}
    for method in METHODS[1:]:
        dif = np.array([r["add_mm"]-baseline[r["scene_id"]] for r in test_rows if r["method"]==method])
        paired[method] = dict(mean_delta_mm=float(dif.mean()), better_count=int((dif<0).sum()), count=len(dif))
    report = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        protocol=dict(methods=METHODS, fixed_delta_m=.006, solve_iterations=25, validation_seeds=[461,462,463], test_seeds=[471,472,473], validation_scenes=18, test_scenes=36, points_per_scene=160, probe_points=256, pose_success="translation <5 mm AND rotation <2 deg; synthetic diagnostic, not benchmark threshold", network_train_steps=0, robot_interactions=0, units="m internally", frame="object to camera", timestamp="each scene independent synthetic t=0", initialization="same unweighted Kabsch", tuning="no hyperparameter search; select method by validation ADD only", validation_hash=val_hash, test_hash=test_hash, ood_hash=ood_hash),
        selected_on_validation=winner, validation=val, test=test, ood_uncertainty_flipped=ood, paired_test_against_l2=paired,
        scope="Five solver heads; not trained multi-scale feature networks, official datasets, or real robot success.")
    (args.out_dir/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir/"per-scene.json").write_text(json.dumps(dict(validation=val_rows,test=test_rows,ood=ood_rows), indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(METHODS))
    ax.bar(x-.18, [test[m]["add_mean_mm"] for m in METHODS], .36, yerr=[test[m]["add_seed_std_mm"] for m in METHODS], capsize=3, label="correct uncertainty")
    ax.bar(x+.18, [ood[m]["add_mean_mm"] for m in METHODS], .36, yerr=[ood[m]["add_seed_std_mm"] for m in METHODS], capsize=3, label="wrong uncertainty, identical points")
    ax.set(xticks=x, xticklabels=METHODS, ylabel="held-out point mean error (mm)", title="Same observations, same budget; error bars = std of 3 seed means")
    ax.legend()
    fig.savefig(args.out_dir/"controlled-comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    for i, method in enumerate(METHODS[1:]):
        vals = [r["add_mm"]-baseline[r["scene_id"]] for r in test_rows if r["method"]==method]
        ax.scatter(np.full(len(vals), i)+np.linspace(-.15,.15,len(vals)), vals, s=12, alpha=.65)
    ax.axhline(0, color="black", linestyle="--")
    ax.set(xticks=np.arange(4), xticklabels=METHODS[1:], ylabel="paired error change vs L2 (mm)", title="Each dot is the same scene under two methods; lower is better")
    fig.savefig(args.out_dir/"paired-effects.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
