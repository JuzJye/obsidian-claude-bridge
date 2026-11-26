#!/usr/bin/env python3
"""Day56: synthetic acceptance ledger, never a robot experiment.

Python 3.11, NumPy, Matplotlib. --corruption injects one ledger defect.
Exit 0 means this audit script ran; read artifact_integrity_pass in results.json.
"""
import argparse
from collections import Counter
import csv
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


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(center-half), float(center+half)]


def fixture():
    rng = np.random.default_rng(56)
    expected = [{"trial_id": f"t{i:03d}", "seed": 5600+i,
                 "initial_xyz_m": np.round(rng.uniform([.3, -.1, .04], [.5, .1, .1]), 6).tolist()}
                for i in range(40)]
    protocol = {"schema": "day56-v1", "source": "preset_synthetic_fixture",
                "frame": "robot_base", "length_unit": "m", "time_unit": "s",
                "success_rule": {"min_lift_m": .08, "min_hold_s": 1., "no_collision": True},
                "timeout_s": 5., "expected_trials": expected}
    rows = []
    reasons = ["perception_failure", "ik_failure", "collision_stop", "capture_failure"]
    for i, trial in enumerate(expected):
        ok = i < 24
        reason = "success" if ok else reasons[(i-24)//4]
        rows.append({**trial, "protocol_sha256": digest(protocol), "frame": "robot_base",
                     "length_unit": "m", "time_unit": "s", "obs_stamp_s": 0.,
                     "command_stamp_s": .02, "end_stamp_s": 2. if ok else 1.,
                     "lift_m": .10 if ok else 0., "hold_s": 1.2 if ok else 0.,
                     "collision": reason == "collision_stop", "success": ok,
                     "terminal_reason": reason})
    return protocol, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corruption", choices=["none", "drop_failures", "duplicate", "wrong_unit",
                                                "false_success", "hash_mismatch"], default="none")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day56")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    protocol, rows = fixture()
    # Seal reference fixture BEFORE fault injection. This is an integrity check, not authentication.
    reference_sha = digest(sorted(rows, key=lambda r: r["trial_id"]))
    if args.corruption == "drop_failures":
        rows = [r for r in rows if r["success"]]
    elif args.corruption == "duplicate":
        rows.append(dict(rows[0]))
    elif args.corruption == "wrong_unit":
        rows[0]["length_unit"] = "mm"
    elif args.corruption == "false_success":
        rows[24]["success"] = True
    elif args.corruption == "hash_mismatch":
        rows[0]["end_stamp_s"] += .1
    expected = {r["trial_id"]: r for r in protocol["expected_trials"]}
    counts = Counter(r["trial_id"] for r in rows)
    missing, extra = sorted(expected.keys()-counts.keys()), sorted(counts.keys()-expected.keys())
    semantic = lambda r: (r["lift_m"] >= .08 and r["hold_s"] >= 1. and not r["collision"])
    checks = {
        "unique_ids": all(n == 1 for n in counts.values()),
        "complete_expected_ids": not missing and not extra,
        "protocol_hash": all(r["protocol_sha256"] == digest(protocol) for r in rows),
        "units_and_frame": all((r["length_unit"], r["time_unit"], r["frame"]) == ("m", "s", "robot_base") for r in rows),
        "initial_state_and_seed": all(r["trial_id"] in expected and all(r[k] == expected[r["trial_id"]][k]
                                      for k in ["seed", "initial_xyz_m"]) for r in rows),
        "time_order_and_deadline": all(0 <= r["obs_stamp_s"] <= r["command_stamp_s"] <= r["end_stamp_s"] <= 5. for r in rows),
        "success_definition": all(r["success"] == semantic(r) and (r["terminal_reason"] == "success") == r["success"] for r in rows),
        "reference_content_hash": digest(sorted(rows, key=lambda r: r["trial_id"])) == reference_sha,
    }
    passed = all(checks.values())
    valid = {r["trial_id"]: r for r in rows if r["trial_id"] in expected and counts[r["trial_id"]] == 1
             and r["success"] == semantic(r) and r["length_unit"] == "m"}
    observed_successes = sum(r["success"] for r in valid.values())
    n = len(expected)
    result = {"source": "preset_synthetic_fixture", "corruption": args.corruption, "checks": checks,
              "artifact_integrity_pass": passed, "project_acceptance": "pending",
              "pending_evidence": ["official_dataset_baseline", "real_or_contact_physics_closed_loop", "learner_assessment"],
              "expected_trials": n, "recorded_rows": len(rows), "missing_ids": missing, "extra_ids": extra,
              "observed_successes": observed_successes,
              "naive_row_success_rate": sum(r["success"] for r in rows)/len(rows),
              "validated_success_rate": observed_successes/n if passed else None,
              "missing_outcome_bounds": [observed_successes/n, (observed_successes+len(missing))/n]
                  if all(checks[k] for k in checks if k not in ["complete_expected_ids", "reference_content_hash"]) else None,
              "wilson_95": wilson(observed_successes, n) if passed else None,
              "terminal_counts": dict(Counter(r["terminal_reason"] for r in rows)),
              "audit_compute_ms": (time.perf_counter()-start)*1000}
    manifest = {"protocol": protocol, "reference_records_sha256": reference_sha,
                "actual_records_sha256": digest(sorted(rows, key=lambda r: r["trial_id"])),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "interaction_budget": {"hardware_trials": 0, "physics_rollouts": 0, "synthetic_records_expected": n}}
    for name, obj in [("manifest", manifest), ("results", result)]:
        (args.out/f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n")
    with (args.out/"ledger.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    labels = ["Observed successes", "Observed failures", "Missing expected IDs"]
    values = [observed_successes, len(valid)-observed_successes, len(missing)]
    ax.bar(labels, values, color=["#178f82", "#d46b45", "#9586ba"])
    for i, v in enumerate(values):
        ax.text(i, v+.5, str(v), ha="center")
    ax.set(ylim=(0, 43), ylabel="Trial count", title=f"Fixed registry N=40 | {args.corruption}")
    ax.axhline(n, ls="--", c="gray", label="Expected denominator = 40")
    ax.legend(loc="upper right")
    fig.savefig(args.out/"ledger_counts.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    lo, hi = wilson(24, 40)
    axes[0].errorbar([.6], [1], xerr=[[.6-lo], [hi-.6]], fmt="o", capsize=7, color="#178f82", label="Complete fixture: 24/40")
    axes[0].scatter([1.], [0], color="#d46b45", marker="x", s=70, label="Success-only: 24/24")
    axes[0].set(xlim=(0, 1.08), ylim=(-.6, 1.8), yticks=[0, 1], yticklabels=["Biased subset", "Full registry"], xlabel="Fraction", title="Counting example; not robot reliability")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].axis("off")
    text = (f"Current audit: {args.corruption}\n\nArtifact integrity: {'PASS' if passed else 'FAIL'}\n"
            "Official-data baseline: PENDING\nContact-physics / robot: PENDING\nLearner assessment: PENDING\n\nProject acceptance: PENDING")
    axes[1].text(.02, .95, text, va="top", fontsize=11, linespacing=1.6,
                 bbox={"boxstyle": "round", "facecolor": "#f1f4f7", "edgecolor": "#a0abb5"})
    fig.suptitle("Wilson interval assumes iid Bernoulli outcomes; fixture outcomes are preset", fontsize=10)
    fig.savefig(args.out/"evidence_gates.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
