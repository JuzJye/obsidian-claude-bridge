"""Day20: geometry integration, time-aligned chunks, and oracle 1-D diffusion.

Synthetic CPU diagnostics, not ACT/Diffusion Policy training or deployment.
The diffusion example knows a two-point data distribution analytically.
"""
from pathlib import Path
import json
import platform
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rigid_fit(x, y):
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((y-yc).T @ (x-xc))
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(u @ vt)
    r = u @ fix @ vt
    return r, yc-r @ xc


def project(p, k):
    q = p @ k.T
    return q[:, :2] / q[:, 2:]


def backproject(uv, z, k):
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(k).T
    return rays * z[:, None]


def temporal_ensemble(chunks, valid, decay):
    length, horizon = chunks.shape
    out = np.zeros(length)
    for t in range(length):
        starts = np.arange(max(0, t-horizon+1), t+1)
        offsets = t-starts
        keep = valid[starts, offsets]
        values = chunks[starts[keep], offsets[keep]]
        # Chronological order: index zero is the OLDEST prediction.
        weights = np.exp(-decay*np.arange(len(values)))
        out[t] = weights @ values / weights.sum()
    return out


def oracle_diffusion(rng, count=2000, steps=120):
    # Data x0 is +1 or -1 with equal probability; amplitude = 0.35*x0 metres.
    beta = np.linspace(0.001, 0.15, steps)
    alpha = 1-beta
    abar = np.cumprod(alpha)
    x = rng.normal(size=count)
    snapshots = {steps: x.copy()}
    for t in range(steps-1, -1, -1):
        a = abar[t]
        # E[x0|xt] for the known equal-weight two-point mixture.
        x0_mean = np.tanh(np.sqrt(a)*x/(1-a))
        eps = (x-np.sqrt(a)*x0_mean)/np.sqrt(1-a)
        mean = (x-beta[t]/np.sqrt(1-a)*eps)/np.sqrt(alpha[t])
        previous = abar[t-1] if t > 0 else 1.
        var = beta[t]*(1-previous)/(1-a)
        x = mean + np.sqrt(var)*rng.normal(size=count) if t > 0 else mean
        if t in (60, 10, 0):
            snapshots[t] = x.copy()
    return x, snapshots, float(abar[-1])


def main():
    out = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day20"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20)
    # Known asymmetric model points; correspondences are supplied, not detected.
    obj = rng.uniform(-1, 1, (80, 3))*[0.06, 0.035, 0.045]
    obj -= obj.mean(0)
    r, _ = cv2.Rodrigues(np.array([0.15, -0.25, 0.4]))
    t = np.array([0.08, -0.03, 0.8])
    cam = obj @ r.T+t
    k = np.array([[600., 0, 320], [0, 610, 240], [0, 0, 1]])
    uv = project(cam, k)
    pixel_noise = rng.normal(0, 0.15, uv.shape)
    measured_uv = uv+pixel_noise
    cam_good = backproject(measured_uv, cam[:, 2], k)
    chain = []
    estimates = []
    tcp_offset = np.array([0.0, 0.0, 0.08])
    for factor in (1.0, 1.05):
        cloud = backproject(measured_uv, factor*cam[:, 2], k)
        rh, th = rigid_fit(obj, cloud)
        fit = obj @ rh.T+th
        grasp = rh @ tcp_offset+th
        truth = r @ tcp_offset+t
        chain.append(dict(depth_factor=factor,
            translation_error_mm=float(1000*np.linalg.norm(th-t)),
            rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(rh@r.T)-1)/2, -1, 1)))),
            fit_rmse_mm=float(1000*np.sqrt(np.mean(np.sum((fit-cloud)**2, axis=1)))),
            grasp_error_mm=float(1000*np.linalg.norm(grasp-truth)),
            cloud_reprojection_rmse_px=float(np.sqrt(np.mean(np.sum((project(cloud, k)-measured_uv)**2, axis=1))))))
        estimates.append(fit)
    ok, rv, tv = cv2.solvePnP(obj, measured_uv, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    pnp_error = float(1000*np.linalg.norm(tv[:, 0]-t))
    assert np.max(np.abs(backproject(uv, cam[:, 2], k)-cam)) < 1e-12

    length, horizon = 70, 8
    target = np.where(np.arange(length) >= 30, 0.06, 0.)
    observation = target+rng.normal(0, 0.008, length)
    chunks = np.repeat(observation[:, None], horizon, axis=1)
    valid = np.arange(length)[:, None]+np.arange(horizon)[None, :] < length
    smooth0 = temporal_ensemble(chunks, valid, 0.)
    smooth6 = temporal_ensemble(chunks, valid, 0.6)
    temporal = []
    for name, pred in (("latest", observation), ("ensemble_m0", smooth0), ("ensemble_m0.6", smooth6)):
        cross = np.flatnonzero(pred[30:] >= 0.03)[0]
        temporal.append(dict(method=name, stationary_rmse_mm=float(1000*np.sqrt(np.mean(pred[8:28]**2))), half_step_delay_samples=int(cross)))
    zeros = temporal_ensemble(np.zeros((4, 2)), np.ones((4, 2), bool), 0.1)
    assert np.all(zeros == 0), "Zero is a valid action; never use nonzero as validity."

    samples, snapshots, end_alpha = oracle_diffusion(np.random.default_rng(2020))
    x_path = np.linspace(0, 1, 301)
    def clearance(amplitudes):
        y = amplitudes[:, None]*np.sin(np.pi*x_path[None, :])
        return np.min(np.sqrt((x_path[None, :]-0.5)**2+y*y), axis=1)-0.18
    mean_clearance = float(clearance(np.array([0.]))[0])
    candidate_clearance = clearance(0.35*samples)
    results = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__, device="CPU"), seed=20,
        geometry=chain, pnp_translation_error_mm=pnp_error, temporal=temporal,
        oracle_diffusion=dict(samples=len(samples), steps=120, final_forward_alpha_bar=end_alpha,
            positive_fraction=float(np.mean(samples > 0)), mean_action_path_clearance_m=mean_clearance,
            sample_min_clearance_m=float(candidate_clearance.min()), sampled_path_collision_fraction=float(np.mean(candidate_clearance < 0)),
            max_abs_mode_error=float(np.max(np.abs(np.abs(samples)-1)))),
        boundary="Provided correspondences; no detector, ACT or DP network trained; oracle scalar diffusion; no robot interaction.")
    (out/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    ax[0].scatter(uv[:, 0], uv[:, 1], s=12, label="ideal projected keypoints")
    ax[0].scatter(measured_uv[:, 0], measured_uv[:, 1], s=8, label="same noisy pixels in both runs")
    ax[0].invert_yaxis(); ax[0].set(xlabel="u (px)", ylabel="v (px)", title="Fixed image evidence")
    ax[0].legend(fontsize=8)
    ax[1].scatter(cam[:, 0], cam[:, 2], s=15, label="true camera points")
    for fac, cloud in zip((1., 1.05), estimates):
        ax[1].scatter(cloud[:, 0], cloud[:, 2], s=10, label=f"rigid fit: depth x {fac}")
    ax[1].set(xlabel="camera x (m)", ylabel="camera z (m)", title="Depth scale changes metric pose")
    ax[1].legend(fontsize=8); fig.savefig(out/"geometry-chain.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for name, pred in (("target", target), ("latest", observation), ("ensemble m=0", smooth0), ("ensemble m=0.6", smooth6)):
        ax[0].plot(np.arange(length)*0.02, pred*1000, label=name)
    ax[0].set(xlabel="physical time (s)", ylabel="position target (mm)", title="Noise reduction and delayed response")
    ax[0].legend(fontsize=8)
    for start in range(24, 34):
        future = np.arange(start, start+horizon)
        ax[1].plot(future, chunks[start]*1000, marker=".", alpha=0.6)
    ax[1].axvline(31, color="black", linestyle="--", label="same execution step")
    ax[1].set(xlabel="execution step", ylabel="predicted target (mm)", title="Ensemble vertically, at identical timestamps")
    ax[1].legend(fontsize=8); fig.savefig(out/"chunk-timing.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for amp, label in ((.35, "upper mode"), (-.35, "lower mode"), (0., "conditional mean")):
        ax[0].plot(x_path, amp*np.sin(np.pi*x_path), label=label)
    ax[0].add_patch(plt.Circle((.5, 0), .18, color="gray", alpha=.35))
    ax[0].set(xlabel="x (m)", ylabel="y (m)", title="Mean of valid paths can hit an obstacle", aspect="equal")
    ax[0].legend(fontsize=8)
    for step in (120, 60, 10, 0):
        ax[1].hist(snapshots[step], bins=np.linspace(-3, 3, 61), density=True, histtype="step", label=f"noise index {step}")
    ax[1].set(xlabel="normalized path amplitude", ylabel="density", title="Known-distribution oracle denoising")
    ax[1].legend(fontsize=8); fig.savefig(out/"multimodal-actions.png", dpi=160); plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
