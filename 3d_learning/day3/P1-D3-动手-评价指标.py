#!/usr/bin/env python3
"""Day 3 synthetic experiments. Python 3.9+, standard library only.
From vault root:
  python3 3d_learning/day3/P1-D3-动手-评价指标.py
No network, files, datasets, or robot commands are used.
Distances are Euclidean (not squared) and measured in mm.
"""
import argparse
import math


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be finite and >= 0")
    return number


def bounded_steps(value):
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("steps must be in [1, 100]")
    return number


def iou_equal_cubes(side, offset):
    """Two equal axis-aligned cubes, displaced only along x."""
    if side <= 0:
        raise ValueError("cube side must be positive")
    intersection = max(side - abs(offset), 0.0) * side * side
    return intersection / (2 * side ** 3 - intersection)


def directed_distance(source, target):
    """Mean nearest-neighbor Euclidean distance; empty clouds undefined."""
    if not source or not target:
        raise ValueError("point clouds must be nonempty")
    return sum(min(math.dist(p, q) for q in target) for p in source) / len(source)


def chamfer(source, target):
    """Sum of two directional means. Unit: mm, NOT mm squared."""
    return directed_distance(source, target) + directed_distance(target, source)


def add(gt, pred):
    """Fixed model-point correspondence after applying respective poses."""
    if not gt or len(gt) != len(pred):
        raise ValueError("ADD needs nonempty, corresponding points")
    return sum(math.dist(p, q) for p, q in zip(gt, pred)) / len(gt)


def ring(radius=50, count=36):
    return [(radius * math.cos(2 * math.pi * i / count),
             radius * math.sin(2 * math.pi * i / count), 0.0)
            for i in range(count)]


def rotate_z(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(c*x-s*y, s*x+c*y, z) for x, y, z in points]


def bc_mse(action):
    return ((action + 1) ** 2 + (action - 1) ** 2) / 2


def rollout(gain, steps):
    """x[t+1] = x[t] + gain * 1, x[0] = 0."""
    return [gain * t for t in range(steps + 1)]


def fit_gain(actions, displacements):
    """Least squares for displacement = k * action, no intercept."""
    if len(actions) != len(displacements) or not actions:
        raise ValueError("need paired training transitions")
    denominator = sum(a*a for a in actions)
    if denominator == 0:
        raise ValueError("training actions must excite the system")
    return sum(a*d for a, d in zip(actions, displacements)) / denominator


def verify():
    """Checks tied to independent analytic examples and metric boundaries."""
    assert math.isclose(iou_equal_cubes(100, 20), 2/3)
    assert math.isclose(iou_equal_cubes(100, 60), 1/4)
    assert iou_equal_cubes(100, 0) == 1
    assert iou_equal_cubes(100, 100) == 0
    assert iou_equal_cubes(100, 200) == 0
    assert iou_equal_cubes(100, -20) == iou_equal_cubes(100, 20)
    small = [(0., 0., 0.), (10., 0., 0.)]
    full = small + [(20., 0., 0.)]
    assert directed_distance(small, full) == 0
    assert math.isclose(directed_distance(full, small), 10/3)
    assert math.isclose(chamfer(full, small), chamfer(small, full))
    points = ring()
    rotated = rotate_z(points, math.pi)
    assert math.isclose(add(points, rotated), 100)
    assert directed_distance(points, rotated) < 1e-10
    shifted = [(x+5, y, z) for x, y, z in points]
    assert math.isclose(add(points, shifted), 5)
    assert bc_mse(0) < bc_mse(-1) == bc_mse(1)
    assert abs(rollout(10, 10)[-1] - rollout(8, 10)[-1]) == 20
    assert fit_gain([-1, .5, 1, 2], [-10, 5, 10, 20]) == 10
    for fn in (lambda: directed_distance([], full),
               lambda: add(small, full), lambda: iou_equal_cubes(0, 1)):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was not rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "iou", "chamfer", "pose", "bc", "world"],
                        default="all")
    parser.add_argument("--side-mm", type=positive, default=100.)
    parser.add_argument("--offset-mm", type=nonnegative, default=20.)
    parser.add_argument("--model-gain", type=nonnegative, default=8.)
    parser.add_argument("--steps", type=bounded_steps, default=10)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verify()
    print("CHECKS PASS: analytic metric examples, symmetry, missing data and rollout.")
    if args.check_only:
        return
    print("All examples are synthetic. Units: mm; Chamfer uses unsquared distances.\n")
    if args.stage in ("all", "iou"):
        print("A | IoU: equal cubes, shift only along x")
        print(f"side={args.side_mm:g} mm, offset={args.offset_mm:g} mm, "
              f"IoU={iou_equal_cubes(args.side_mm, args.offset_mm):.6f}")
        print("Think: same absolute error, smaller object -> what changes?\n")
    if args.stage in ("all", "chamfer"):
        p = [(0., 0., 0.), (10., 0., 0.)]
        q = p + [(20., 0., 0.)]
        print("B | Missing point: prediction P={0,10}, GT Q={0,10,20} mm")
        print(f"P->Q={directed_distance(p,q):.6f} mm; "
              f"Q->P={directed_distance(q,p):.6f} mm; CD={chamfer(p,q):.6f} mm")
        print("Think: can perfect precision establish full coverage?\n")
    if args.stage in ("all", "pose"):
        gt = ring()
        pred = rotate_z(gt, math.pi)
        print("C | Symmetry: radius 50 mm ring, rotated 180 degrees")
        print(f"ADD={add(gt,pred):.6f} mm; ADD-S={directed_distance(gt,pred):.6f} mm")
        print("Think: does adding a unique handle preserve the same symmetry?\n")
    if args.stage in ("all", "bc"):
        print("D | BC: equally likely expert actions -1 and +1")
        for action in (-1, 0, 1):
            print(f"constant action={action:+d}: MSE={bc_mse(action):.3f}")
        print("Think: the lowest MSE action may point into an obstacle.\n")
    if args.stage in ("all", "world"):
        true = rollout(10, args.steps)
        pred = rollout(args.model_gain, args.steps)
        print("E | Action-conditioned toy model: x'=x+k*a, action=1")
        print("step | true x (mm) | model x (mm) | absolute error (mm)")
        for t in sorted({0, 1, min(5, args.steps), args.steps}):
            print(f"{t:4d} | {true[t]:11.3f} | {pred[t]:12.3f} | {abs(true[t]-pred[t]):.3f}")
        learned = fit_gain([-1, .5, 1, 2], [-10, 5, 10, 20])
        print(f"Gain fitted to four exact synthetic training transitions: {learned:.3f}")
        print("The selected model gain illustrates misspecification. A noiseless fit")
        print("does not establish robustness to contact, noise or unseen actions.")


if __name__ == "__main__":
    main()
