"""Day25: hash features, projected Gaussian splats, and rollout evaluation.

CPU teaching experiments, not the official Instant-NGP or 3DGS implementation.
The dynamics have state [position(m), velocity(m/s)], command acceleration(m/s^2),
dt=0.1s and actuator clipping at +/-1m/s^2. No robot or network access.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import torch
from torch import nn


class Hash2D(nn.Module):
    """Four bilinearly interpolated feature grids, capped by a hash table."""
    def __init__(self, capacity, features=2):
        super().__init__()
        self.resolutions = [8, 16, 32, 64]
        self.capacity = capacity
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(min((n + 1)**2, capacity), features))
            for n in self.resolutions])
        for table in self.tables:
            nn.init.uniform_(table, -1e-3, 1e-3)
        self.head = nn.Sequential(nn.Linear(4 * features, 32), nn.ReLU(),
                                  nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, xy):
        encoded = []
        for n, table in zip(self.resolutions, self.tables):
            scaled = xy.clamp(0., 1. - 1e-7) * n
            lower = scaled.floor().long()
            fraction = scaled - lower
            value = torch.zeros((len(xy), table.shape[1]), dtype=xy.dtype)
            for dx, dy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                corner = lower + lower.new_tensor([dx, dy])
                if (n + 1)**2 <= self.capacity:
                    index = corner[:, 0] + (n + 1) * corner[:, 1]
                else:
                    index = (corner[:, 0] ^ (corner[:, 1] * 2654435761)) % len(table)
                wx = fraction[:, 0] if dx else 1. - fraction[:, 0]
                wy = fraction[:, 1] if dy else 1. - fraction[:, 1]
                value = value + (wx * wy)[:, None] * table[index]
            encoded.append(value)
        return self.head(torch.cat(encoded, dim=-1))[:, 0]


def signal(xy):
    x, y = xy[:, 0], xy[:, 1]
    return .5 + .2 * torch.sin(8 * torch.pi * x) * torch.sin(8 * torch.pi * y) + .2 * torch.cos(2 * torch.pi * x)


def fit_hash(steps, seed):
    torch.manual_seed(seed)
    train = torch.rand(4096, 2)
    target = signal(train)
    axis = (torch.arange(64, dtype=torch.float32) + .5) / 64
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    test = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    truth = signal(test)
    arrays = {'hash_truth': truth.numpy().reshape(64, 64)}
    records = {}
    for capacity in [16, 512]:
        torch.manual_seed(seed + 1)
        model = Hash2D(capacity)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        batch_generator = torch.Generator().manual_seed(seed + 2)
        losses = []
        start = time.perf_counter()
        for i in range(steps):
            ids = torch.randint(len(train), (512,), generator=batch_generator)
            loss = (model(train[ids]) - target[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0 or i == steps - 1:
                losses.append([i + 1, loss.item()])
        with torch.no_grad():
            prediction = model(test)
            rmse = torch.sqrt((prediction - truth).square().mean()).item()
        arrays[f'hash_{capacity}'] = prediction.numpy().reshape(64, 64)
        arrays[f'loss_{capacity}'] = np.array(losses)
        records[str(capacity)] = {
            'test_rmse': rmse, 'encoding_parameters': sum(t.numel() for t in model.tables),
            'all_parameters': sum(p.numel() for p in model.parameters()),
            'training_seconds': time.perf_counter() - start}
    return records, arrays


def rotation_z(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def project_gaussian(mean, covariance, focal=80., center=32.):
    x, y, z = mean
    if z <= 0:
        raise ValueError('Gaussian center must be in front of the camera')
    uv = np.array([focal * x / z + center, focal * y / z + center])
    jacobian = np.array([[focal/z, 0., -focal*x/z**2],
                         [0., focal/z, -focal*y/z**2]])
    return uv, jacobian @ covariance @ jacobian.T


def splat_demo():
    yy, xx = np.mgrid[:64, :64]
    pixels = np.stack([xx, yy], axis=-1)
    means = [np.array([0., 0., 1.5]), np.array([0., 0., 2.])]
    colors = [np.array([1., 0., 0.]), np.array([0., 0., 1.])]
    opacities = [.75, .6]
    alphas, covariances = [], []
    for mean, angle, opacity in zip(means, [35., -35.], opacities):
        rotation = rotation_z(angle)
        covariance = rotation @ np.diag(np.array([.15, .035, .04])**2) @ rotation.T
        uv, cov2 = project_gaussian(mean, covariance)
        covariances.append(cov2)
        offset = pixels - uv
        squared = np.einsum('...i,ij,...j->...', offset, np.linalg.inv(cov2), offset)
        alphas.append(opacity * np.exp(-.5 * squared))
    def composite(order):
        rgb = np.zeros((64, 64, 3))
        transmittance = np.ones((64, 64))
        for i in order:
            rgb += (transmittance * alphas[i])[..., None] * colors[i]
            transmittance *= 1 - alphas[i]
        return rgb + transmittance[..., None]  # White background.
    correct, reversed_order = composite([0, 1]), composite([1, 0])
    assert np.allclose(correct[32, 32], [.85, .1, .25])
    assert np.allclose(reversed_order[32, 32], [.4, .1, .7])
    return {'front_to_back_center_rgb': correct[32, 32].tolist(),
            'reversed_center_rgb': reversed_order[32, 32].tolist(),
            'image_mean_absolute_difference': float(abs(correct-reversed_order).mean()),
            'covariance_eigenvalues_px2': [np.linalg.eigvalsh(c).tolist() for c in covariances]}, {
            'splat_correct': correct, 'splat_wrong_order': reversed_order,
            'splat_alpha_red': alphas[0], 'splat_alpha_blue': alphas[1]}


def true_step(state, command, rng, noise_std=.008):
    p, v = state[..., 0], state[..., 1]
    velocity = .94 * v + .1 * (np.clip(command, -1., 1.) - .6 * p)
    velocity = velocity + rng.normal(0., noise_std, size=velocity.shape)
    return np.stack([p + .1 * velocity, velocity], axis=-1)


def trajectories(rng, count, length, amplitude):
    states = np.empty((count, length + 1, 2))
    states[:, 0] = rng.uniform([-.4, -.2], [.4, .2], size=(count, 2))
    # Constant commands per episode expose the unobserved actuator saturation.
    commands = rng.uniform(-amplitude, amplitude, size=(count, 1))
    commands = np.broadcast_to(commands, (count, length)).copy()
    for k in range(length):
        states[:, k+1] = true_step(states[:, k], commands[:, k], rng)
    return states, commands


def features(state, action):
    return np.concatenate([state, action[..., None], np.ones_like(action[..., None])], axis=-1)


def dynamics_demo(seed):
    training, actions = trajectories(np.random.default_rng(seed), 300, 30, .6)
    x = features(training[:, :-1], actions).reshape(-1, 4)
    y = training[:, 1:].reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    # Controlled parameter perturbation; this is not another fitted model.
    biased = coef.copy()
    biased[1, 1] += .04
    biased[1, 0] += .004
    summary, arrays = {'coefficient_rows_p_v_a_bias': coef.tolist()}, {}
    for distribution, amplitude in [('id', .6), ('ood', 2.)]:
        states, commands = trajectories(np.random.default_rng(seed + 50), 200, 10, amplitude)
        arrays[f'{distribution}_truth'] = states
        arrays[f'{distribution}_actions'] = commands
        summary[distribution] = {}
        for label, matrix, clip_input in [('learned', coef, False), ('biased', biased, False),
                                          ('known_limit', coef, True)]:
            inputs = np.clip(commands, -1., 1.) if clip_input else commands
            teacher = features(states[:, :-1], inputs) @ matrix
            rollout = np.empty_like(states)
            rollout[:, 0] = states[:, 0]
            for k in range(10):
                rollout[:, k+1] = features(rollout[:, k], inputs[:, k]) @ matrix
            teacher_rmse = np.sqrt(np.mean((teacher-states[:, 1:])**2, axis=0))
            rollout_rmse = np.sqrt(np.mean((rollout[:, 1:]-states[:, 1:])**2, axis=0))
            summary[distribution][label] = {
                'teacher_position_rmse_m_at_1_5_10': teacher_rmse[[0, 4, 9], 0].tolist(),
                'free_position_rmse_m_at_1_5_10': rollout_rmse[[0, 4, 9], 0].tolist(),
                'free_velocity_rmse_mps_at_1_5_10': rollout_rmse[[0, 4, 9], 1].tolist()}
            arrays[f'{distribution}_{label}_teacher_rmse'] = teacher_rmse
            arrays[f'{distribution}_{label}_free_rmse'] = rollout_rmse
            arrays[f'{distribution}_{label}_rollout'] = rollout
        summary[distribution]['saturated_command_fraction'] = float((abs(commands)>1.).mean())
    assert np.allclose(arrays['id_learned_free_rmse'][0], arrays['id_learned_teacher_rmse'][0])
    return summary, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day25')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--seed', type=int, default=25)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    hash_result, arrays = fit_hash(args.steps, args.seed)
    splat_result, extra = splat_demo()
    arrays.update(extra)
    dynamic_result, extra = dynamics_demo(args.seed)
    arrays.update(extra)
    results = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
               'torch': torch.__version__, 'device': 'cpu', 'threads': 1},
               'seed': args.seed, 'hash_steps': args.steps, 'hash': hash_result,
               'splat': splat_result, 'dynamics': dynamic_result,
               'total_seconds_excluding_plot': time.perf_counter()-start,
               'scope': 'Synthetic mechanisms only; no official NGP/3DGS, real data or robot.'}
    np.savez_compressed(args.output/'arrays.npz', **arrays)
    (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
