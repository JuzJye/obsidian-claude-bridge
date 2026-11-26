"""D44: robust rigid registration losses, CPU NumPy; no model training."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rigid(x, y, w):
    w = w / w.sum()
    xc, yc = w @ x, w @ y
    u, _, vt = np.linalg.svd((x - xc).T @ ((y - yc) * w[:, None]))
    d = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    r = vt.T @ d @ u.T
    return r, yc - r @ xc


def fit(x, y, kind, delta=.008, steps=30):
    r, t = rigid(x, y, np.ones(len(x)))
    for _ in range(steps):
        residual = np.linalg.norm(x @ r.T + t - y, axis=1)
        w = np.ones(len(x))
        if kind == "Huber":
            w = np.minimum(1., delta / np.maximum(residual, 1e-12))
        elif kind == "Cauchy":
            w = 1. / (1. + (residual / delta) ** 2)
        r, t = rigid(x, y, w)
    return r, t, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier-fraction", type=float, default=.2)
    ap.add_argument("--delta-mm", type=float, default=8.)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "_assets/3d_learning/day44")
    args = ap.parse_args()
    if not 0 <= args.outlier_fraction < 1 or args.delta_mm <= 0:
        raise ValueError("fraction must be in [0,1), delta positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(44)
    x = rng.uniform(-1, 1, (160, 3)) * [.07, .05, .04]
    theta = .35
    r_true = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    t_true = np.array([.03, -.01, .65])
    truth = x @ r_true.T + t_true
    y_clean = truth + rng.normal(0, .002, truth.shape)
    y = y_clean.copy()
    bad = rng.permutation(len(x))[:int(len(x) * args.outlier_fraction)]
    y[bad] += [.07, -.04, .06]
    held_x = rng.uniform(-1, 1, (400, 3)) * [.07, .05, .04]
    held_true = held_x @ r_true.T + t_true
    records, estimates = {}, {}
    for name in ["L2", "Huber", "Cauchy"]:
        start = time.perf_counter()
        r, t, w = fit(x, y, name, args.delta_mm / 1000)
        elapsed = (time.perf_counter() - start) * 1000
        pred = held_x @ r.T + t
        err = np.linalg.norm(pred - held_true, axis=1)
        records[name] = dict(translation_mm=float(np.linalg.norm(t - t_true) * 1000),
            rotation_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(r @ r_true.T) - 1) / 2, -1, 1)))),
            held_add_mm=float(err.mean() * 1000), observed_rmse_mm=float(np.sqrt(np.mean(np.sum((x @ r.T + t - y) ** 2, 1))) * 1000),
            inlier_weight=float(np.mean(w[np.setdiff1d(np.arange(len(x)), bad)])),
            outlier_weight=None if not len(bad) else float(w[bad].mean()), fit_ms=elapsed)
        estimates[name] = (r, t, w)
    # Wholly clean negative control; same samples, no injected outliers.
    clean = {}
    for name in records:
        r, t, _ = fit(x, y_clean, name, args.delta_mm / 1000)
        clean[name] = float(np.linalg.norm(held_x @ r.T + t - held_true, axis=1).mean() * 1000)
    report = dict(seed=44, data="synthetic metric correspondences; fixed object, independent held-out points", units="m; outputs mm and deg", frame="object to camera", timestamp="synthetic single instant t=0", initial_state="unweighted Kabsch for every method", train_samples=160, heldout_samples=400, optimizer_steps=30, network_train_steps=0, robot_interactions=0,
        outlier_fraction=args.outlier_fraction, delta_mm=args.delta_mm, environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__), corrupted=records, clean_held_add_mm=clean,
        scope="Known correspondences, rigid object; no correspondence network, NOCS/EPro-PnP reproduction, or robot execution.")
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    u = np.linspace(-5, 5, 400)
    ax[0].plot(u, .5 * u*u, label="L2")
    ax[0].plot(u, np.where(np.abs(u) <= 1, .5*u*u, np.abs(u)-.5), label="Huber")
    ax[0].plot(u, .5*np.log1p(u*u), label="Cauchy")
    ax[0].set(xlabel="normalized residual r / delta", ylabel="loss / delta^2", title="Same near-zero curvature, different tails")
    ax[0].legend()
    pos = np.arange(3)
    ax[1].bar(pos-.18, [records[k]["held_add_mm"] for k in records], .36, label="20% biased outliers" if args.outlier_fraction == .2 else f"{args.outlier_fraction:.0%} outliers")
    ax[1].bar(pos+.18, list(clean.values()), .36, label="clean control")
    ax[1].set(xticks=pos, xticklabels=list(records), ylabel="held-out mean point error (mm)", title="Evaluate on clean held-out geometry")
    ax[1].legend()
    fig.savefig(args.out_dir / "robust-losses.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    order = np.r_[np.setdiff1d(np.arange(len(x)), bad), bad]
    for name, (_, _, w) in estimates.items():
        ax.plot(w[order], label=name)
    ax.axvline(len(x)-len(bad)-.5, color="black", linestyle="--", label="inlier / outlier split (evaluation only)")
    ax.set(xlabel="correspondence index after display-only sorting", ylabel="IRLS weight", ylim=(-.05, 1.1), title="Robust weights are not calibrated confidence")
    ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "influence-weights.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
