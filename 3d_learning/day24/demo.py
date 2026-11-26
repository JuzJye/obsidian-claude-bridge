"""Day 24: CPU neural radiance field and independently split one-step dynamics.

Run from the vault root: python -B 3d_learning/day24/demo.py
Only synthetic data; no downloads, pretrained weights, or hardware access.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ASSETS = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day24"
NEAR, FAR = 0.6, 3.0  # metres along a unit camera ray


def camera_rays(azimuth_deg, size=28):
    """OpenCV camera axes: right, down, forward; c2w maps camera to world."""
    theta = math.radians(float(azimuth_deg))
    origin = torch.tensor([1.8 * math.cos(theta), 1.8 * math.sin(theta), 0.45])
    forward = F.normalize(-origin, dim=0)
    right = F.normalize(torch.linalg.cross(forward, torch.tensor([0., 0., 1.])), dim=0)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward], dim=1)
    v, u = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    focal = size * 1.1
    dirs_cam = torch.stack([(u - (size - 1) / 2) / focal,
                            (v - (size - 1) / 2) / focal, torch.ones_like(u)], -1)
    directions = F.normalize(dirs_cam.float() @ rotation.T, dim=-1).reshape(-1, 3)
    return origin.expand_as(directions), directions


def encoding(x, levels):
    # Includes raw coordinates. Frequencies follow the paper's pi convention.
    features = [x]
    for k in range(levels):
        features.extend([torch.sin((2 ** k) * math.pi * x),
                         torch.cos((2 ** k) * math.pi * x)])
    return torch.cat(features, dim=-1)


class SmallNeRF(nn.Module):
    def __init__(self, width=48, levels=4):
        super().__init__()
        self.levels = levels
        self.trunk = nn.Sequential(nn.Linear(3 * (1 + 2 * levels), width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.density = nn.Linear(width, 1)
        self.color = nn.Sequential(nn.Linear(width + 3, width), nn.ReLU(),
                                   nn.Linear(width, 3))
        nn.init.constant_(self.density.bias, -1.)

    def forward(self, points, directions):
        h = self.trunk(encoding(points, self.levels))
        support = (points.abs().amax(dim=-1) <= 0.75).float()
        sigma = F.softplus(self.density(h)[..., 0]) * support
        rgb = torch.sigmoid(self.color(torch.cat([h, directions], dim=-1)))
        return sigma, rgb


def analytic_field(points, directions):
    """Known continuous density and color, not a mesh or a path-traced dataset."""
    centers = points.new_tensor([[-.23, -.06, -.02], [.21, .04, .02], [0., .03, .30]])
    radii = points.new_tensor([[.19, .24, .24], [.18, .21, .20], [.15, .16, .14]])
    q = ((points[..., None, :] - centers) / radii).square().sum(-1)
    lobes = 24. * torch.exp(-2. * q)
    support = (points.abs().amax(dim=-1) <= .75).float()
    sigma = lobes.sum(-1) * support
    colors = points.new_tensor([[.92, .16, .10], [.08, .36, .95], [.12, .86, .22]])
    rgb = (lobes[..., None] * colors).sum(-2) / (lobes.sum(-1, keepdim=True) + 1e-8)
    stripe = .80 + .20 * torch.cos(18. * points[..., 2:3])
    # A modest direction-dependent term is part of the synthetic target.
    rgb = (rgb * stripe + .04 * directions[..., 2:3].square()).clamp(0., 1.)
    return sigma, rgb


def render(field, origins, directions, samples=48, stratified=False):
    """Finite, equal-width bins; each query represents its whole bin.

    Unit rays make t and delta lengths in metres. We use bin widths (not
    distances between jittered query locations), a deliberate quadrature choice.
    """
    count = len(origins)
    offsets = torch.rand(count, samples) if stratified else torch.full((count, samples), .5)
    delta = (FAR - NEAR) / samples
    t = NEAR + (torch.arange(samples)[None] + offsets) * delta
    points = origins[:, None, :] + directions[:, None, :] * t[..., None]
    dirs = directions[:, None, :].expand_as(points)
    sigma, colors = field(points.reshape(-1, 3), dirs.reshape(-1, 3))
    sigma, colors = sigma.reshape(count, samples), colors.reshape(count, samples, 3)
    optical = sigma * delta
    alpha = -torch.expm1(-optical)
    prefix = torch.cat([torch.zeros(count, 1), optical.cumsum(-1)[:, :-1]], -1)
    transmittance = torch.exp(-prefix)
    weights = transmittance * alpha
    background_weight = torch.exp(-optical.sum(-1))
    rgb = (weights[..., None] * colors).sum(1) + background_weight[:, None]
    opacity = weights.sum(-1)
    conditional_depth = (weights * t).sum(-1) / opacity.clamp_min(1e-8)
    return rgb, opacity, conditional_depth


@torch.no_grad()
def render_batches(field, origins, directions, samples=48, chunk=256):
    pieces = [render(field, origins[i:i + chunk], directions[i:i + chunk], samples)
              for i in range(0, len(origins), chunk)]
    return tuple(torch.cat([part[k] for part in pieces]) for k in range(3))


def psnr(prediction, target):
    return float(-10. * torch.log10((prediction - target).square().mean().clamp_min(1e-12)))


def fit_radiance_field(steps, size, samples):
    torch.manual_seed(24)
    torch.set_num_threads(2)
    train_angles = np.arange(0., 360., 30.)
    val_angles = np.array([15., 135.])
    test_angles = np.array([75., 255.])
    angles = np.concatenate([train_angles, val_angles, test_angles])
    splits = np.array([0] * 12 + [1] * 2 + [2] * 2)
    rays = [camera_rays(angle, size) for angle in angles]
    origins = torch.stack([pair[0] for pair in rays])
    directions = torch.stack([pair[1] for pair in rays])
    truths = [render_batches(analytic_field, o, d, samples=128) for o, d in rays]
    target = torch.stack([v[0] for v in truths])
    target_opacity = torch.stack([v[1] for v in truths])
    field = SmallNeRF()
    optimizer = torch.optim.Adam(field.parameters(), lr=2e-3)
    train_o, train_d = origins[:12].reshape(-1, 3), directions[:12].reshape(-1, 3)
    train_y = target[:12].reshape(-1, 3)
    history = []
    checkpoints = {0, steps // 4, steps // 2, 3 * steps // 4, steps}
    for step in range(steps + 1):
        if step in checkpoints:
            val_pred = torch.stack([render_batches(field, *rays[k], samples=samples)[0]
                                    for k in [12, 13]])
            history.append([step, psnr(val_pred, target[12:14])])
            print(f"step={step:4d} validation_psnr={history[-1][1]:.3f} dB", flush=True)
        if step == steps:
            break
        indices = torch.randint(len(train_o), (96,))
        prediction = render(field, train_o[indices], train_d[indices], samples, stratified=True)[0]
        loss = F.mse_loss(prediction, train_y[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = 2e-3 * (0.2 ** ((step + 1) / steps))
    # All image metrics use the final, fixed-budget checkpoint; no test selection.
    predicted = torch.stack([render_batches(field, o, d, samples=samples)[0] for o, d in rays])
    per_view = [psnr(predicted[k], target[k]) for k in range(len(angles))]
    sample_sweep = {}
    for n in [12, 24, 48, 96]:
        pred = torch.stack([render_batches(field, *rays[k], samples=n)[0] for k in [14, 15]])
        sample_sweep[str(n)] = psnr(pred, target[14:16])
    foreground = target_opacity[14:16] > .1
    data = dict(angles=angles, split=splits, origins=origins.numpy(), directions=directions.numpy(),
                target=target.numpy(), predicted=predicted.numpy(), target_opacity=target_opacity.numpy(),
                validation_history=np.array(history), per_view_psnr=np.array(per_view),
                size=np.array(size), samples=np.array(samples), near=np.array(NEAR), far=np.array(FAR))
    np.savez_compressed(ASSETS / "radiance.npz", **data)
    torch.save({"state_dict": field.state_dict(), "width": 48, "levels": 4,
                "near": NEAR, "far": FAR, "samples": samples, "size": size}, ASSETS / "small_nerf.pt")
    return dict(steps=steps, ray_batch=96, samples=samples, size=size,
                parameters=sum(p.numel() for p in field.parameters()), seed=24,
                train_views=train_angles.tolist(), validation_views=val_angles.tolist(),
                test_views=test_angles.tolist(), target_samples=128,
                train_psnr=psnr(predicted[:12], target[:12]),
                validation_psnr=psnr(predicted[12:14], target[12:14]),
                test_psnr=psnr(predicted[14:16], target[14:16]),
                test_foreground_psnr=psnr(predicted[14:16][foreground], target[14:16][foreground]),
                per_view_psnr=per_view, fixed_model_samples_test_psnr=sample_sweep,
                validation_history=history)


def true_step(state, action, dt=.1, noise=0.):
    """Command acceleration is clipped at +/-1 m/s^2; scalar or array input."""
    p, v = np.asarray(state)[..., 0], np.asarray(state)[..., 1]
    v_next = .94 * v + dt * (np.clip(action, -1., 1.) - .6 * p) + noise
    return np.stack([p + dt * v_next, v_next], axis=-1)


def design_matrix(state, action, mode="full"):
    state, action = np.asarray(state), np.asarray(action)
    if action.shape == state.shape[:-1]:
        action = action[..., None]
    ones = np.ones_like(action)
    if mode == "full":
        return np.concatenate([state, action, ones], axis=-1)
    if mode == "position_only":
        return np.concatenate([state[..., :1], action, ones], axis=-1)
    if mode == "no_action":
        return np.concatenate([state, ones], axis=-1)
    raise ValueError(mode)


def dynamics_experiment():
    rng = np.random.default_rng(2401)
    dt, noise_std = .1, .008
    states = np.empty((120, 40, 2))
    actions = np.empty((120, 40, 1))
    next_states = np.empty_like(states)
    for trajectory in range(120):
        state = rng.uniform([-.6, -.4], [.6, .4])
        action = 0.
        for step in range(40):
            if step % 5 == 0:
                action = rng.uniform(-.6, .6)
            next_state = true_step(state, action, dt, rng.normal(0., noise_std))
            states[trajectory, step] = state
            actions[trajectory, step, 0] = action
            next_states[trajectory, step] = next_state
            state = next_state
    split = np.array([0] * 80 + [1] * 20 + [2] * 20)
    report, coefficients, predictions = {}, {}, {}
    for mode in ["full", "position_only", "no_action"]:
        design = design_matrix(states, actions, mode)
        coef = np.linalg.lstsq(design[split == 0].reshape(-1, design.shape[-1]),
                               next_states[split == 0].reshape(-1, 2), rcond=None)[0]
        pred = design @ coef
        coefficients[f"coef_{mode}"] = coef
        predictions[f"pred_{mode}"] = pred
        report[mode] = {}
        for name, value in [("train", 0), ("validation", 1), ("test", 2)]:
            rmse = np.sqrt(np.mean((pred[split == value] - next_states[split == value]) ** 2,
                                   axis=(0, 1)))
            report[mode][name] = {"position_rmse_m": float(rmse[0]),
                                   "velocity_rmse_m_s": float(rmse[1])}
    np.savez_compressed(ASSETS / "dynamics.npz", states=states, actions=actions,
                        next_states=next_states, trajectory_ids=np.arange(120), split=split,
                        dt=np.array(dt), noise_std=np.array(noise_std),
                        **coefficients, **predictions)
    report.update(seed=2401, dt_s=dt, velocity_noise_std_m_s=noise_std,
                  trajectories={"train": 80, "validation": 20, "test": 20},
                  transitions_per_trajectory=40, command_acceleration_range_m_s2=[-.6, .6],
                  actuator_limit_m_s2=1., coef_full=coefficients["coef_full"].tolist())
    return report


def hand_check():
    sigma = np.array([2., 3.])
    delta = np.array([.2, .2])
    alpha = -np.expm1(-sigma * delta)
    transmittance = np.exp(-np.r_[0., np.cumsum(sigma * delta)[:-1]])
    weights = transmittance * alpha
    background = np.exp(-np.sum(sigma * delta))
    color = weights @ np.array([[1., 0., 0.], [0., 0., 1.]]) + background * np.ones(3)
    assert np.isclose(weights.sum() + background, 1.)
    # Reparameterizing a ray cannot change physical opacity.
    correct = 1. - np.exp(-2. * .2)
    missing_direction_norm = 1. - np.exp(-2. * .1)
    return dict(alpha=alpha.tolist(), transmittance=transmittance.tolist(),
                weights=weights.tolist(), background_weight=float(background),
                white_background_rgb=color.tolist(),
                correct_alpha=float(correct), missing_norm_alpha=float(missing_direction_norm))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    if args.steps < 1 or args.size < 4 or args.samples < 2:
        parser.error("steps>=1, size>=4 and samples>=2 are required")
    ASSETS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = {"verified_date": "2026-09-29", "device": "cpu", "python": platform.python_version(),
              "numpy": np.__version__, "torch": torch.__version__, "hand_check": hand_check()}
    report["dynamics"] = dynamics_experiment()
    print("Saved independent dynamics trajectories and fitted coefficients.", flush=True)
    report["radiance"] = fit_radiance_field(args.steps, args.size, args.samples)
    report["elapsed_s"] = time.perf_counter() - start
    (ASSETS / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
