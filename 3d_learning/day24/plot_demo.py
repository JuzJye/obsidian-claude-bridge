"""Plot saved Day24 evidence; no training and no external resources."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/day24-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo import ASSETS, SmallNeRF, analytic_field, NEAR, FAR


def main():
    z = np.load(ASSETS / "radiance.npz")
    report = json.loads((ASSETS / "results.json").read_text())
    size = int(z["size"])
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for row, k in enumerate([12, 14]):
        target = z["target"][k].reshape(size, size, 3)
        prediction = z["predicted"][k].reshape(size, size, 3)
        error = np.abs(prediction - target).mean(-1)
        ax[row, 0].imshow(target)
        ax[row, 0].set_title(f"{'Validation' if row == 0 else 'Test'} {z['angles'][k]:.0f} deg: target")
        ax[row, 1].imshow(prediction)
        ax[row, 1].set_title(f"Final model: {z['per_view_psnr'][k]:.2f} dB")
        heat = ax[row, 2].imshow(error, vmin=0., vmax=.12, cmap="magma")
        ax[row, 2].set_title("Mean absolute RGB error")
        ax[row, 3].imshow(z["target_opacity"][k].reshape(size, size), vmin=0, vmax=1, cmap="gray")
        ax[row, 3].set_title("Target opacity (not a label)")
        for a in ax[row]:
            a.set_xticks([])
            a.set_yticks([])
    fig.colorbar(heat, ax=ax[:, 2], shrink=.8, label="RGB [0, 1] error")
    fig.suptitle("A learned radiance field: entire held-out camera views, 28 x 28 pixels", fontsize=14)
    fig.savefig(ASSETS / "heldout_views.png", dpi=160)
    plt.close(fig)

    saved = torch.load(ASSETS / "small_nerf.pt", map_location="cpu", weights_only=True)
    field = SmallNeRF(saved["width"], saved["levels"])
    field.load_state_dict(saved["state_dict"])
    field.eval()
    t = NEAR + (torch.arange(48) + .5) * (FAR - NEAR) / 48
    k, pixel = 14, size * (size // 2) + size // 2
    o = torch.from_numpy(z["origins"][k, pixel])
    d = torch.from_numpy(z["directions"][k, pixel])
    points, dirs = o[None] + t[:, None] * d[None], d.expand(48, 3)
    with torch.no_grad():
        truth_sigma = analytic_field(points, dirs)[0].numpy()
        fitted_sigma = field(points, dirs)[0].numpy()
    def weights(sigma):
        optical = sigma * (FAR - NEAR) / 48
        return np.exp(-np.r_[0., np.cumsum(optical)[:-1]]) * -np.expm1(-optical)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    hist = z["validation_history"]
    ax[0, 0].plot(hist[:, 0], hist[:, 1], "o-", color="#147d92")
    ax[0, 0].set(xlabel="Optimization step", ylabel="Validation PSNR (dB)",
                 title="Only training-view RGB enters the optimizer")
    sweep = report["radiance"]["fixed_model_samples_test_psnr"]
    ax[0, 1].plot(list(map(int, sweep)), list(sweep.values()), "o-", color="#b05b16")
    ax[0, 1].set(xlabel="Samples per ray (same final model)", ylabel="Test PSNR (dB)",
                 title="Single-variable quadrature comparison")
    ax[0, 1].set_xticks([12, 24, 48, 96])
    ax[1, 0].plot(t, truth_sigma, label="Analytic target", color="#147d92")
    ax[1, 0].plot(t, fitted_sigma, "--", label="Learned", color="#cb6744")
    ax[1, 0].set(xlabel="Distance along unit ray (m)", ylabel="Density (1/m)",
                 title="RGB fitting does not uniquely recover density")
    ax[1, 1].plot(t, weights(truth_sigma), label="Analytic target", color="#147d92")
    ax[1, 1].plot(t, weights(fitted_sigma), "--", label="Learned", color="#cb6744")
    ax[1, 1].set(xlabel="Distance along unit ray (m)", ylabel="Weight T_i * alpha_i",
                 title="Visibility gates a sample's RGB contribution")
    for a in ax.flat:
        a.grid(alpha=.2)
    ax[1, 0].legend()
    ax[1, 1].legend()
    fig.suptitle("Optimization, sampling, and one held-out central ray", fontsize=14)
    fig.savefig(ASSETS / "rendering_evidence.png", dpi=160)
    plt.close(fig)

    dyn = np.load(ASSETS / "dynamics.npz")
    truth = dyn["next_states"][100]
    times = (np.arange(40) + 1) * float(dyn["dt"])
    modes = ["full", "no_action", "position_only"]
    names = ["p, v, a", "p, v", "p, a"]
    colors = ["#147d92", "#b05b16", "#a1447d"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax[0, 0].plot(times, truth[:, 0], "k-", lw=2, label="Observed next position")
    for mode, name, color in zip(modes, names, colors):
        pred = dyn["pred_" + mode][100]
        ax[0, 0].plot(times, pred[:, 0], "--", label=f"Input {name}", color=color)
        ax[0, 1].plot(times, 1000 * (pred[:, 0] - truth[:, 0]), label=f"Input {name}", color=color)
    ax[0, 0].set(xlabel="Time (s)", ylabel="Position (m)", title="One independent test trajectory")
    ax[0, 1].set(xlabel="Time (s)", ylabel="One-step position error (mm)", title="Missing state or action has a cost")
    for axis, key, unit in [(ax[1, 0], "position_rmse_m", "Position RMSE (mm)"),
                             (ax[1, 1], "velocity_rmse_m_s", "Velocity RMSE (mm/s)")]:
        vals = [1000 * report["dynamics"][m]["test"][key] for m in modes]
        axis.bar(names, vals, color=colors)
        axis.set(ylabel=unit, xlabel="Model inputs", title="All 800 held-out test transitions")
        for i, value in enumerate(vals):
            axis.text(i, value, f"{value:.3f}", ha="center", va="bottom")
        axis.set_ylim(0, max(vals) * 1.16)
    ax[0, 0].legend(fontsize=8)
    for a in ax.flat:
        a.grid(axis="y", alpha=.2)
    fig.suptitle("Every prediction uses observed s_t: this is not a free rollout", fontsize=14)
    fig.savefig(ASSETS / "one_step_dynamics.png", dpi=160)
    plt.close(fig)
    print("Saved heldout_views.png, rendering_evidence.png, one_step_dynamics.png")


if __name__ == "__main__":
    main()
