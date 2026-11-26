"""Optional plot regeneration: Python 3.10+ and matplotlib.

Run from any directory: python3 plot_demo.py --output /tmp/day5-figures
Core experiments in demo.py do not depend on matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo import track, error_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    colors = {"ff": "#d47b22", "p": "#3777b0", "ff+p": "#178077"}
    labels = {"ff": "Feedforward only", "p": "P feedback only",
              "ff+p": "Feedforward + P"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for mode in colors:
        rows = track(mode)
        t = [r["t_s"] for r in rows]
        axes[0].plot(t, [1000*r["position_m"] for r in rows],
                     color=colors[mode], label=labels[mode], lw=2)
        axes[1].plot(t, [1000*r["error_m"] for r in rows],
                     color=colors[mode], label=labels[mode], lw=2)
    axes[0].plot(t, [1000*r["reference_m"] for r in rows],
                 color="#333333", ls="--", label="Reference", lw=1.6)
    axes[0].set(title="A. Follow the same timed reference", ylabel="Position [mm]")
    axes[1].set(title="B. Error = reference - actual", ylabel="Tracking error [mm]")
    for ax in axes:
        ax.axvline(1.0, color="#777777", ls=":", label="-15 mm disturbance")
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8.5)
    fig.suptitle("Teaching plant: actual velocity = 0.8 x command; Kp = 8 /s; dt = 0.02 s")
    fig.savefig(args.output / "tracking.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for kp, color in [(10.0, "#178077"), (45.0, "#b64951")]:
        errors = error_sequence(kp)
        axes[0].semilogy([i*0.05 for i in range(len(errors))],
                         [max(abs(e), 1e-15) for e in errors],
                         color=color, lw=2, label=f"Kp={kp:g}/s; Kp*dt={kp*0.05:g}")
    for delay, color in [(0, "#178077"), (3, "#b64951")]:
        errors = error_sequence(10, delay_steps=delay)
        axes[1].plot([i*0.05 for i in range(len(errors))], errors,
                     color=color, lw=2, label=f"Delay={delay} steps ({delay*0.05:.2f} s)")
    axes[0].set(title="C. No delay: gain versus convergence", ylabel="Absolute error [m], log scale")
    axes[1].set(title="D. Same gain, different sensing delay", ylabel="Signed error [m]")
    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    fig.suptitle("Separate ideal model: no saturation, unit plant gain; dt = 0.05 s")
    fig.savefig(args.output / "sampled-stability.png")
    plt.close(fig)
    print(f"Saved tracking.png and sampled-stability.png; matplotlib {matplotlib.__version__}")


if __name__ == "__main__":
    main()
