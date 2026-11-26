#!/usr/bin/env python3
"""Day 7: rotation, QR fitting and a planar sensor-control simulator.

Standard library only. Point robot, velocity command, no contact dynamics.
All positions in metres, velocity in m/s, rotation angles in radians.
"""
import argparse
import json
import math
import platform
import random
import time
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return list(map(list, zip(*a)))


def matmul(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def rotation_vector(w):
    theta = math.sqrt(dot(w, w))
    x, y, z = w
    skew = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    square = matmul(skew, skew)
    if theta < 1e-8:
        aa, bb = 1 - theta * theta / 6, 0.5 - theta * theta / 24
    else:
        aa, bb = math.sin(theta) / theta, (1 - math.cos(theta)) / theta**2
    return [[float(i == j) + aa * skew[i][j] + bb * square[i][j]
             for j in range(3)] for i in range(3)]


def qr_fit(design, target):
    # Modified Gram-Schmidt for a small full-column-rank teaching example.
    columns = transpose(design)
    q = []
    count = len(columns)
    r = [[0.0] * count for _ in range(count)]
    for j, col in enumerate(columns):
        v = col[:]
        for i, qi in enumerate(q):
            r[i][j] = dot(qi, v)
            v = [a - r[i][j] * b for a, b in zip(v, qi)]
        r[j][j] = math.sqrt(dot(v, v))
        if r[j][j] <= 1e-12 * max(1.0, math.sqrt(dot(col, col))):
            raise ValueError("columns are dependent or numerically unresolved")
        q.append([a / r[j][j] for a in v])
    qty = [dot(qi, target) for qi in q]
    beta = [0.0] * count
    for j in reversed(range(count)):
        beta[j] = (qty[j] - sum(r[j][k] * beta[k]
                               for k in range(j + 1, count))) / r[j][j]
    residual = [y - yp for y, yp in zip(target, matvec(design, beta))]
    return {"beta": beta, "Q": transpose(q), "R": r, "residual": residual,
            "residual_norm": math.sqrt(dot(residual, residual))}


def segment_distance(a, b, center):
    delta = [y - x for x, y in zip(a, b)]
    length2 = dot(delta, delta)
    fraction = 0.0 if length2 == 0 else max(
        0.0, min(1.0, dot([c - x for c, x in zip(center, a)], delta) / length2))
    closest = [x + fraction * d for x, d in zip(a, delta)]
    return math.dist(closest, center)


class PlanarReach:
    """reset/step-style teaching interface; not a registered Gymnasium Env."""
    dt, max_speed, threshold, max_steps = 0.05, 0.35, 0.008, 120
    goal = [0.4, 0.3]
    camera_yaw = math.pi / 2
    obstacle_center, obstacle_radius = [0.2, 0.15], 0.05

    def __init__(self, obstacle=False):
        self.obstacle = obstacle
        self.finished = True

    def observation(self):
        # Camera-frame target displacement; the policy does not read world p.
        world_error = [g - p for g, p in zip(self.goal, self.position)]
        c, s = math.cos(self.camera_yaw), math.sin(self.camera_yaw)
        return [c * world_error[0] + s * world_error[1],
                -s * world_error[0] + c * world_error[1]]

    def reset(self, seed=7):
        rng = random.Random(seed)
        self.position = [rng.uniform(-0.02, 0.02) for _ in range(2)]
        self.steps, self.finished = 0, False
        return self.observation(), {"seed": seed}

    def step(self, velocity):
        if self.finished:
            raise RuntimeError("reset is required before another step")
        if len(velocity) != 2 or not all(math.isfinite(v) for v in velocity):
            raise ValueError("velocity must contain two finite values")
        norm = math.sqrt(dot(velocity, velocity))
        scale = min(1.0, self.max_speed / max(norm, 1e-12))
        actual = [v * scale for v in velocity]
        old = self.position[:]
        self.position = [p + self.dt * v for p, v in zip(old, actual)]
        self.steps += 1
        collision = self.obstacle and segment_distance(
            old, self.position, self.obstacle_center) <= self.obstacle_radius
        distance = math.dist(self.position, self.goal)
        outside = any(abs(p) > 1 for p in self.position)
        success = distance < self.threshold and not collision and not outside
        terminated = success or collision or outside
        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        reason = ("success" if success else "collision" if collision else
                  "workspace" if outside else "timeout" if truncated else "running")
        # Dimensionless cost uses a reference length of 1 metre.
        reward = -distance / 1.0 + float(success) - float(collision)
        info = {"distance_m": distance, "reason": reason, "success": success,
                "executed_velocity_m_s": actual}
        return self.observation(), reward, terminated, truncated, info


def episode(seed, yaw_bias_deg=0.0, obstacle=False):
    env = PlanarReach(obstacle)
    obs, _ = env.reset(seed)
    records = [{"step": 0, "position_m": env.position[:],
                "distance_m": math.dist(env.position, env.goal)}]
    controller_times = []
    while not env.finished:
        start = time.perf_counter()
        yaw = env.camera_yaw + math.radians(yaw_bias_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        velocity = [2 * (c * obs[0] - s * obs[1]),
                    2 * (s * obs[0] + c * obs[1])]
        controller_times.append(time.perf_counter() - start)
        obs, reward, terminated, truncated, info = env.step(velocity)
        records.append({"step": env.steps, "position_m": env.position[:],
                        "distance_m": info["distance_m"], "reward": reward})
    return {"seed": seed, "yaw_bias_deg": yaw_bias_deg, "obstacle": obstacle,
            "success": info["success"], "reason": info["reason"],
            "terminated": terminated, "truncated": truncated, "steps": env.steps,
            "controller_mean_us": 1e6 * sum(controller_times) / len(controller_times),
            "records": records}


def run(seed=7):
    rotation = rotation_vector([0, 0, math.pi / 2])
    point = [0.2, 0, 0.6]
    translation = [0.4, 0.1, 0]
    base = [a + b for a, b in zip(matvec(rotation, point), translation)]
    recovered = matvec(transpose(rotation), [a - b for a, b in zip(base, translation)])
    assert math.dist(recovered, point) < 1e-12
    fit = qr_fit([[1, 0], [1, 1], [1, 2]], [1, 2, 2])
    assert math.dist(fit["beta"], [7 / 6, 0.5]) < 1e-12
    assert max(abs(x) for x in matvec([[1, 1, 1], [0, 1, 2]],
                                     fit["residual"])) < 1e-12
    try:
        qr_fit([[1, 1], [1, 1], [1, 1]], [1, 2, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("rank-deficient data must be rejected")
    assert segment_distance([0, 0], [1, 0], [0.5, 0]) == 0
    experiments = {}
    for name, bias, obstacle in [("correct_frame", 0, False),
                                 ("yaw_error_90deg", 90, False),
                                 ("straight_line_obstacle", 0, True)]:
        episodes = [episode(seed + i, bias, obstacle) for i in range(5)]
        experiments[name] = {"successes": sum(e["success"] for e in episodes),
                             "attempts": len(episodes),
                             "interaction_steps": sum(e["steps"] for e in episodes),
                             "episodes": episodes}
    return {"environment": {"python": platform.python_version(), "os": platform.system(),
                            "seed_start": seed, "data": "synthetic", "device": "CPU"},
            "geometry": {"R_base_camera": rotation, "t_base_camera_m": translation,
                         "point_camera_m": point, "point_base_m": base,
                         "round_trip_error_m": math.dist(recovered, point)},
            "qr": fit, "simulation": experiments, "checks": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Environment:", report["environment"])
    print("Base point:", report["geometry"]["point_base_m"])
    print("QR beta:", report["qr"]["beta"], "residual norm:", report["qr"]["residual_norm"])
    for name, case in report["simulation"].items():
        print(name, f"{case['successes']}/{case['attempts']}",
              "steps:", case["interaction_steps"],
              "reasons:", [e["reason"] for e in case["episodes"]])
    print("Numerical checks:", report["checks"])
