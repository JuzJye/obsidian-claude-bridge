"""Day 5: action contracts, time scaling, tracking, and sampled-data delay.

Python 3.10+, standard library, deterministic CPU examples; no robot I/O.
Run: python3 demo.py --csv /tmp/day5-tracking.csv
All positions are in meters, times in seconds, and velocities in m/s.
"""
import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Packet:
    action_norm: tuple[float, float] = (0.5, 0.0)
    frame: str = "tcp"
    mode: str = "delta_position"
    decoded_unit: str = "m"
    period_s: float = 0.05
    observation_time_s: float = 9.95
    command_time_s: float = 9.98


def rotate_xy(vector, yaw):
    x, y = vector
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def decode_delta(packet, now_s=10.0, tcp_yaw=math.pi / 2):
    """Teaching contract: normalized 2-D TCP translation, fixed statistics.

    Returns a base-frame displacement. This is not a collision checker,
    driver, complete VLA adapter, or permission to execute physical motion.
    """
    if packet.frame != "tcp" or packet.mode != "delta_position":
        raise ValueError("unsupported frame or action mode")
    if packet.decoded_unit != "m":
        raise ValueError("decoded translation must use meters")
    timing = (now_s, packet.period_s, packet.observation_time_s,
              packet.command_time_s, tcp_yaw)
    if not all(math.isfinite(v) for v in timing):
        raise ValueError("non-finite metadata")
    if not math.isclose(packet.period_s, 0.05, abs_tol=1e-9):
        raise ValueError("action period differs from this contract")
    if not packet.observation_time_s <= packet.command_time_s <= now_s:
        raise ValueError("timestamp order invalid")
    if now_s - packet.observation_time_s > 0.10:
        raise ValueError("observation is stale")
    if len(packet.action_norm) != 2 or not all(
        math.isfinite(v) and abs(v) <= 1 for v in packet.action_norm
    ):
        raise ValueError("expected two finite normalized values in [-1,1]")
    # Teaching statistics: mu=(0,0) m, sigma=(0.02,0.02) m.
    local = tuple(0.02 * value for value in packet.action_norm)
    return rotate_xy(local, tcp_yaw)


def cubic(t, duration=1.5, goal=0.1):
    """Return position and analytic velocity for a rest-to-rest cubic."""
    if t <= 0:
        return 0.0, 0.0
    if t >= duration:
        return goal, 0.0
    s = t / duration
    return goal * (3*s*s - 2*s*s*s), goal/duration * (6*s - 6*s*s)


def track(mode, kp=8.0, delay_steps=0, dt=0.02, total_s=3.0):
    """Velocity plant x_next=x+0.8*u*dt, with one -0.015 m disturbance.

    Feedforward uses exact average reference velocity over each interval,
    so its final bias is not an Euler integration artifact.
    The actuator has a +/-0.2 m/s limit. Delayed sensing uses held initial
    state before the start; the command itself has no extra queue delay.
    """
    if mode not in {"ff", "p", "ff+p"}:
        raise ValueError("unknown tracking mode")
    if kp < 0 or delay_steps < 0:
        raise ValueError("kp and delay_steps must be nonnegative")
    history = [0.0]
    rows = [dict(mode=mode, t_s=0.0, reference_m=0.0, position_m=0.0,
                 error_m=0.0, command_mps=0.0, saturated=0)]
    for i in range(round(total_s / dt)):
        t, tn = i * dt, (i + 1) * dt
        ref, _ = cubic(t)
        ref_next, _ = cubic(tn)
        observed = history[max(0, len(history)-1-delay_steps)]
        u = (ref_next-ref)/dt if mode in {"ff", "ff+p"} else 0.0
        if mode in {"p", "ff+p"}:
            u += kp * (ref-observed)
        limited = max(-0.2, min(0.2, u))
        x = history[-1] + 0.8 * limited * dt
        if i + 1 == round(1.0 / dt):
            x -= 0.015
        history.append(x)
        rows.append(dict(mode=mode, t_s=tn, reference_m=ref_next,
                         position_m=x, error_m=ref_next-x,
                         command_mps=limited, saturated=int(abs(u) > 0.2)))
    return rows


def error_sequence(kp, dt=0.05, delay_steps=0, steps=40):
    """Unsaturated ideal velocity plant; fixed target, initial error 0.1 m.

    e[n+1] = e[n] - kp*dt*e[n-delay]. The initial prehistory is constant.
    Separate from track(): no gain mismatch, no velocity limit, no impulse.
    """
    errors = [0.1]
    for n in range(steps):
        errors.append(errors[-1] - kp*dt*errors[max(0, n-delay_steps)])
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.kp) or args.kp < 0 or args.delay_steps < 0:
        parser.error("kp must be finite/nonnegative; delay-steps nonnegative")

    delta = decode_delta(Packet())
    print(f"Frame: TCP delta (0.010,0.000) m -> base ({delta[0]:.3f},{delta[1]:.3f}) m")
    print("Target: base (0.300,0.200) -> (0.300,0.210); wrong-frame target (0.310,0.200)")
    bad_packets = {
        "stale": replace(Packet(), observation_time_s=9.7),
        "unit": replace(Packet(), decoded_unit="mm"),
        "mode": replace(Packet(), mode="velocity"),
        "future": replace(Packet(), command_time_s=10.1),
    }
    rejected = 0
    for name, packet in bad_packets.items():
        try:
            decode_delta(packet)
        except ValueError as exc:
            rejected += 1
            print(f"Contract rejects {name}: {exc}")

    distance, vmax, amax = 0.1, 0.1, 0.4
    duration = max(1.5*distance/vmax, math.sqrt(6*distance/amax))
    print(f"Cubic minimum duration under stated limits: {duration:.3f} s")
    all_rows = []
    results = {}
    for mode in ("ff", "p", "ff+p"):
        rows = track(mode, args.kp, args.delay_steps)
        results[mode] = rows
        all_rows.extend(rows)
        rms = math.sqrt(sum(r["error_m"]**2 for r in rows)/len(rows))
        sat = sum(r["saturated"] for r in rows)
        print(f"Tracking {mode:4s}: final_error={rows[-1]['error_m']:.8f} m, "
              f"RMSE={rms:.8f} m, saturated_steps={sat}/150")

    for label, kp, delay in (("stable", 10.0, 0),
                              ("excess_gain", 45.0, 0),
                              ("delayed", 10.0, 3)):
        errors = error_sequence(kp, delay_steps=delay)
        print(f"Stability {label:11s}: kp*dt={kp*0.05:.2f}, delay={delay}, "
              f"final_error={errors[-1]:.6f} m, max_abs={max(map(abs, errors)):.6f} m")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved tracking CSV: {args.csv}")

    assert math.dist(delta, (0.0, 0.01)) < 1e-12 and rejected == 4
    assert abs(duration - 1.5) < 1e-12
    assert abs(results["ff"][-1]["error_m"] - 0.035) < 1e-12
    assert abs(error_sequence(10.0)[-1]) < 1e-10
    assert abs(error_sequence(45.0)[-1]) > 0.1
    assert max(map(abs, error_sequence(10.0, delay_steps=3))) > 0.1
    print("Analytic contract/trajectory/stability checks: PASS.")
    print("Tracking with user-selected gains is measured, not assumed successful.")


if __name__ == "__main__":
    main()
