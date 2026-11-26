#!/usr/bin/env python3
"""Day 6: attention, action quantization, fitted dynamics, RGB-D -> BEV.

Python standard library only. Synthetic data; no trained VLA or real robot.
Coordinates in metres, velocity in m/s, acceleration command in m/s^2.
"""
import argparse
import json
import math
import platform
import random
from pathlib import Path


def attention(temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    query, keys = [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
    values = [[0.2, 0.1], [0.6, 0.3]]
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(2)
              / temperature for key in keys]
    exps = [math.exp(s - max(scores)) for s in scores]
    weights = [e / sum(exps) for e in exps]
    context = [sum(w * value[d] for w, value in zip(weights, values))
               for d in range(2)]
    return {"weights": weights, "context_m": context}


def quantize(action, low=-0.04, high=0.04, bins=8):
    if not math.isfinite(action) or bins <= 0 or high <= low:
        raise ValueError("invalid quantizer inputs")
    width = (high - low) / bins
    token = min(bins - 1, max(0, math.floor((action - low) / width)))
    decoded = low + (token + 0.5) * width
    return {"input_m": action, "token": token, "decoded_m": decoded,
            "abs_error_m": abs(action - decoded)}


def transition(state, action, rho=0.82, gain=1.2, dt=0.1):
    x, v = state
    next_v = rho * v + gain * action * dt
    return x + next_v * dt, next_v


def fit_dynamics(rows, dt=0.1):
    # Learn v_next = rho * v + gain * (a * dt), using two-column OLS.
    xx = xy = yy = xb = yb = 0.0
    for velocity, action, next_velocity in rows:
        x, y = velocity, action * dt
        xx += x * x
        xy += x * y
        yy += y * y
        xb += x * next_velocity
        yb += y * next_velocity
    determinant = xx * yy - xy * xy
    if xx * yy <= 0 or determinant <= 1e-12 * xx * yy:
        raise ValueError("training inputs lack independent excitation")
    return ((xb * yy - yb * xy) / determinant,
            (yb * xx - xb * xy) / determinant)


def dynamics_report(seed, rho_bias):
    if not math.isfinite(rho_bias):
        raise ValueError("rho_bias must be finite")
    rng = random.Random(seed)
    rows = []
    for _ in range(200):
        velocity, action = rng.uniform(-0.2, 0.2), rng.uniform(-0.5, 0.5)
        _, next_velocity = transition((0.0, velocity), action)
        rows.append((velocity, action, next_velocity))
    rho, gain = fit_dynamics(rows)
    true = fitted = biased = (0.0, 0.2)
    records = []
    for step in range(1, 11):
        action = 0.4 * math.sin(0.8 * step)
        one_step = transition(true, action, rho + rho_bias, gain)
        true = transition(true, action)
        fitted = transition(fitted, action, rho, gain)
        biased = transition(biased, action, rho + rho_bias, gain)
        records.append({"step": step, "action_m_s2": action,
                        "true_x_m": true[0], "fit_x_m": fitted[0],
                        "biased_x_m": biased[0],
                        "teacher_forced_error_m": abs(one_step[0] - true[0]),
                        "rollout_error_m": abs(biased[0] - true[0])})
    minus = transition((0.2, 0.0), -0.5)[0]
    plus = transition((0.2, 0.0), 0.5)[0]
    actionless_rmse = math.sqrt(((minus - 0.2) ** 2 + (plus - 0.2) ** 2) / 2)
    assert abs(rho - 0.82) < 1e-10 and abs(gain - 1.2) < 1e-10
    assert max(abs(r["true_x_m"] - r["fit_x_m"]) for r in records) < 1e-10
    try:
        fit_dynamics([(0.1, 0.0, 0.082)] * 10)
    except ValueError:
        pass
    else:
        raise AssertionError("unidentifiable gain must be rejected")
    return {"rho_fit": rho, "gain_fit": gain, "rho_bias": rho_bias,
            "paired_next_x_m": [minus, plus],
            "actionless_rmse_m": actionless_rmse, "rollout": records}


def synthetic_points():
    # Optical Z-depth, fx=fy=4 px, cx=cy=1.5 px; table camera height=1 m.
    # R_table_camera = diag(1, -1, -1), translation=(0, 0, 1) m.
    depth = [[1, 1, 1, 1], [1, 0.7, 0.7, 1],
             [1, 0.7, 0, 1], [float("nan"), 1, 1, 1]]
    points = []
    for v, row in enumerate(depth):
        for u, z in enumerate(row):
            if not math.isfinite(z) or z <= 0:
                continue
            points.append(((u - 1.5) * z / 4, -(v - 1.5) * z / 4, 1 - z))
    return points


def bev(points, resolution=0.1, low=-0.5, high=0.5):
    # Square XY extent; floor mapping, lower inclusive and upper exclusive.
    if resolution <= 0 or high <= low:
        raise ValueError("invalid grid extent")
    size_float = (high - low) / resolution
    size = round(size_float)
    if not math.isclose(size_float, size):
        raise ValueError("extent must contain an integer number of cells")
    counts = [[0 for _ in range(size)] for _ in range(size)]
    heights = [[None for _ in range(size)] for _ in range(size)]
    for x, y, z in points:
        if not all(math.isfinite(q) for q in (x, y, z)):
            continue
        if not (low <= x < high and low <= y < high):
            continue
        j = min(size - 1, math.floor((x - low) / resolution))
        i = min(size - 1, math.floor((y - low) / resolution))
        counts[i][j] += 1
        heights[i][j] = z if heights[i][j] is None else max(heights[i][j], z)
    return {"counts": counts, "max_height_m": heights,
            "observed": [[c > 0 for c in row] for row in counts],
            "extent_m": [low, high, low, high], "resolution_m": resolution}


def run(seed=7, temperature=1.0, rho_bias=-0.08):
    points = synthetic_points()
    grid = bev(points)
    assert len(points) == 14 and sum(map(sum, grid["counts"])) == 14
    boundary = bev([(-0.5, -0.5, 0.0), (0.5, 0.0, 0.0)])
    assert sum(map(sum, boundary["counts"])) == 1
    assert boundary["counts"][0][0] == 1
    low_object, tall_object = bev([(0, 0, 0.02)]), bev([(0, 0, 0.4)])
    assert low_object["counts"] == tall_object["counts"]
    assert low_object["max_height_m"] != tall_object["max_height_m"]
    quantization = [quantize(a) for a in (0.013, 0.0, 0.08)]
    assert all(quantize(-0.04 + i * 0.0001)["abs_error_m"] <= 0.005 + 1e-12
               for i in range(801))
    attn = attention(temperature)
    assert abs(sum(attn["weights"]) - 1) < 1e-12
    return {"environment": {"python": platform.python_version(),
                            "system": platform.system(), "seed": seed},
            "attention": attn, "quantization": quantization,
            "dynamics": dynamics_report(seed, rho_bias), "bev": grid,
            "points_table_m": points, "checks": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rho-bias", type=float, default=-0.08)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.temperature, args.rho_bias)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
