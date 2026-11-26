"""Day57: synthetic tactile clocks, pairing, arrival-time causality; CPU only.

Run from the vault root. All frequencies, delays and pressures are teaching
choices, not specifications of a commercial sensor. No model is trained.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pressure(t):
    """Non-periodic latent contact pressure, in synthetic kPa."""
    y = np.zeros_like(t, dtype=float)
    for center, width, peak in [(6, .4, 1.3), (13, .23, 1.0),
                                (25, .55, 1.7), (40, .18, .9), (54, .3, 1.4)]:
        y += peak * np.exp(-.5 * ((t - center) / width) ** 2)
    return y


def nearest(mapped, reference):
    hi = np.clip(np.searchsorted(mapped, reference), 1, len(mapped) - 1)
    lo = hi - 1
    return np.where(abs(mapped[lo] - reference) <= abs(mapped[hi] - reference), lo, hi)


def summarize(index, valid, mapped, tv, tt, arrival, decision, pv, pt):
    # Truth is used ONLY for evaluation, never to select a matched sample.
    dt = tt[index] - tv
    correct = (pt[index] >= .4) == (pv >= .4)
    n, k = len(tv), int(valid.sum())
    return {
        "reference_count": n, "valid_count": k, "missing_count": n - k,
        "coverage": k / n,
        "physical_mae_ms_valid": float(np.mean(abs(dt[valid])) * 1000) if k else None,
        "physical_p95_abs_ms_valid": float(np.quantile(abs(dt[valid]), .95) * 1000) if k else None,
        "contact_correct_count": int((correct & valid).sum()),
        "contact_agreement_valid": float(correct[valid].mean()) if k else None,
        "correct_and_covered_fraction_all": float((correct & valid).sum() / n),
        "estimated_future_count_valid": int((valid & (mapped[index] > tv + 1e-12)).sum()),
        "physical_future_count_valid": int((valid & (dt > 1e-12)).sum()),
        "unavailable_at_decision_count_valid": int((valid & (arrival[index] > decision)).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drift-ppm", type=float, default=300.)
    ap.add_argument("--seed", type=int, default=57)
    ap.add_argument("--out", type=Path, default=Path("_assets/3d_learning/day57/default"))
    args = ap.parse_args()
    if not np.isfinite(args.drift_ppm) or abs(args.drift_ppm) > 10000:
        ap.error("this teaching experiment requires finite drift within +/-10000 ppm")
    args.out.mkdir(parents=True, exist_ok=True)
    tic = perf_counter()
    rng = np.random.default_rng(args.seed)
    a, b = 1 + args.drift_ppm * 1e-6, .075
    # Independent synchronous pulses, identified by trigger ID, BEFORE evaluation.
    cal_ref = np.array([-20., -15., -10., -5., 0.])
    cal_raw = a * cal_ref + b + rng.normal(0, 20e-6, len(cal_ref))
    alpha, beta = np.linalg.lstsq(np.c_[cal_raw, np.ones(5)], cal_ref, rcond=None)[0]
    offset = np.mean(cal_raw - cal_ref)
    tv = 1 + np.arange(1800) / 30  # physical/reference clock; vision is the anchor
    tt_all = .002 + np.arange(6200) / 100
    keep = ~(((tt_all >= 23) & (tt_all < 23.4)) |
             ((tt_all >= 45) & (tt_all < 45.12)))
    tt = tt_all[keep]
    raw = a * tt + b
    touch_arrival = tt + .012 + rng.uniform(0, .012, len(tt))
    touch_arrival += .09 * ((tt >= 35) & (tt < 35.3))
    vision_arrival = tv + .020 + rng.uniform(0, .010, len(tv))
    decision = tv + .030  # the current visual sample has arrived by this instant
    pv, pt = pressure(tv), pressure(tt)
    # Calibrated pressure-grid proxy, not an optical tactile RGB image or real force.
    gy, gx = np.mgrid[-1:1:4j, -1:1:4j]
    spatial = np.exp(-((gx[None] - .3 * np.sin(.2 * tt[:, None, None])) ** 2 +
                       gy[None] ** 2) / .4)
    spatial /= spatial.max(axis=(1, 2), keepdims=True)
    taxels = (pt[:, None, None] * spatial).astype(np.float32)
    mappings = {"raw_offline": raw, "offset_offline": raw - offset,
                "affine_offline": alpha * raw + beta}
    pairs, metrics = {}, {}
    for name, mapped in mappings.items():
        index = nearest(mapped, tv)
        valid = abs(mapped[index] - tv) <= .025
        pairs[name] = (index, valid)
        metrics[name] = summarize(index, valid, mapped, tv, tt, touch_arrival, decision, pv, pt)
    mapped = mappings["affine_offline"]
    index, valid = np.zeros(len(tv), dtype=int), np.zeros(len(tv), dtype=bool)
    for i, (ref, now) in enumerate(zip(tv, decision)):
        eligible = np.flatnonzero((mapped <= ref) & (touch_arrival <= now))
        if len(eligible):
            index[i] = eligible[-1]  # latest capture, not latest packet arrival
            valid[i] = ref - mapped[index[i]] <= .025
    pairs["affine_online"] = (index, valid)
    metrics["affine_online"] = summarize(index, valid, mapped, tv, tt, touch_arrival, decision, pv, pt)
    assert np.all(vision_arrival <= decision)
    assert metrics["affine_online"]["estimated_future_count_valid"] == 0
    assert metrics["affine_online"]["unavailable_at_decision_count_valid"] == 0
    assert max(cal_ref) < min(tv)
    report = {
        "data_kind": "synthetic_nonperiodic_contact_and_clock_fixture",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "seed": args.seed, "drift_ppm": args.drift_ppm, "truth_a": a, "truth_b_s": b,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fit_alpha": float(alpha), "fit_beta_s": float(beta), "offset_only_s": float(offset),
        "calibration_max_abs_residual_us": float(abs(alpha * cal_raw + beta - cal_ref).max() * 1e6),
        "calibration_interval_s": [-20, 0], "evaluation_interval_s": [1, float(tv[-1])],
        "sampling_hz": {"vision": 30, "touch": 100}, "dropped_touch_count": int((~keep).sum()),
        "online_max_age_s": .025, "decision_after_vision_capture_s": .030,
        "touch_record_count": len(tt), "contact_threshold_kpa": .4,
        "metrics": metrics, "robot_attempt_count": 0, "training_interaction_count": 0,
        "limits": "No hardware, physical contact simulation, modality-response lag or neural network. "
                  "True physical times are evaluation labels only; calibrator uses separate pulse pairs.",
    }
    arrays = dict(vision_capture_s=tv, vision_arrival_s=vision_arrival, decision_s=decision,
                  touch_truth_s=tt, touch_raw_s=raw, touch_mapped_s=mapped,
                  touch_arrival_s=touch_arrival, touch_taxels_kpa=taxels,
                  vision_contact_proxy_kpa=pv, calibration_reference_s=cal_ref,
                  calibration_device_s=cal_raw)
    for name, (ix, mask) in pairs.items():
        arrays[name + "_index"] = np.where(mask, ix, -1)
        arrays[name + "_valid"] = mask
    np.savez_compressed(args.out / "streams.npz", **arrays)
    colors = ["#a33a2b", "#b18419", "#2066a8", "#23845b"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for (name, (ix, mask)), color in zip(pairs.items(), colors):
        axes[0].plot(tv, np.where(mask, (tt[ix] - tv) * 1000, np.nan),
                     label=name, color=color, lw=1)
        zoom = (tv > 53.1) & (tv < 54.9)
        axes[1].plot(tv[zoom], np.where(mask, pt[ix], np.nan)[zoom], label=name, color=color)
    axes[0].axhline(0, color="black", lw=.6)
    axes[0].set(xlabel="Vision reference time [s]", ylabel="Physical pair residual [ms]",
                title=f"Pairing error against hidden truth: drift {args.drift_ppm:g} ppm; gaps are invalid")
    axes[0].legend(ncol=2)
    axes[1].plot(tv[zoom], pv[zoom], "k--", label="latent pressure at reference")
    axes[1].axhline(.4, color="gray", ls=":", label="contact threshold")
    axes[1].set(xlabel="Vision reference time [s]", ylabel="Synthetic pressure [kPa]",
                title="Same non-periodic event, different timestamp mappings")
    axes[1].legend(ncol=2, fontsize=8)
    names = list(pairs)
    axes[2].bar(np.arange(4) - .18, [metrics[n]["coverage"] for n in names], .36, label="covered / all")
    axes[2].bar(np.arange(4) + .18, [metrics[n]["correct_and_covered_fraction_all"] for n in names],
                .36, label="correct and covered / all")
    axes[2].set(xticks=np.arange(4), xticklabels=names, ylim=(0, 1.06),
                ylabel="Fraction of all 1800 references", title="Coverage remains visible in the denominator")
    axes[2].legend(loc="lower right")
    fig.savefig(args.out / "clock_and_pairing.png", dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.7, 1])
    ax = fig.add_subplot(grid[0, :])
    pick = int(np.argmin(abs(tv - 35.16)))
    local = (tt > 34.94) & (tt < 35.4)
    for cap, arr in zip(tt[local], touch_arrival[local]):
        ax.plot([cap, arr], [0, 1], color="#bbb", lw=.6)
    ax.scatter(tt[local], np.zeros(local.sum()), s=15, label="physical capture", color="#2066a8")
    ax.scatter(touch_arrival[local], np.ones(local.sum()), s=15, label="arrival", color="#a33a2b")
    ax.axvline(tv[pick], color="black", ls="--", label="visual reference")
    ax.axvline(decision[pick], color="#23845b", ls="--", label="decision deadline")
    for name, color in [("affine_offline", "#b18419"), ("affine_online", "#23845b")]:
        j, ok = pairs[name][0][pick], pairs[name][1][pick]
        ax.scatter([tt[j]], [0], s=120, facecolors="none", edgecolors=color,
                   linewidths=2, label=f"{name}: {'valid' if ok else 'age rejected'}")
    ax.set(xlabel="Reference-clock time [s]", yticks=[0, 1], yticklabels=["capture", "arrival"],
           title="A past capture can arrive after the decision: burst transport delay")
    ax.legend(ncol=3, fontsize=8)
    for col, sec in enumerate([52.8, 53.7, 54.]):
        ax = fig.add_subplot(grid[1, col])
        j = np.argmin(abs(tt - sec))
        im = ax.imshow(taxels[j], vmin=0, vmax=1.4, cmap="magma", origin="lower")
        ax.set(title=f"4 x 4 synthetic taxels\nt = {tt[j]:.3f} s", xlabel="taxel column", ylabel="taxel row")
        fig.colorbar(im, ax=ax, label="kPa (synthetic)")
    fig.savefig(args.out / "arrival_and_taxels.png", dpi=160)
    plt.close(fig)
    report["wall_time_s_including_figures"] = perf_counter() - tic
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
