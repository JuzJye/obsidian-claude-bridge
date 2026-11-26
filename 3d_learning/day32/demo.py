"""Conditional flow matching on a two-mode action sketch; CPU, synthetic data."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/day32-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/day32-cache')
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / '_assets/3d_learning/day32'


def data(n, generator):
    goal = (2 * torch.randint(0, 2, (n, 1), generator=generator) - 1).float()
    mode = (2 * torch.randint(0, 2, (n, 1), generator=generator) - 1).float()
    # Two physical coordinates, expressed as dimensionless values using 1 m.
    action = torch.cat([.70 * mode, .60 * goal], 1)
    action += .06 * torch.randn(n, 2, generator=generator)
    return goal, action


def sample(model, noise, goal, steps):
    x = noise.clone()
    with torch.no_grad():
        for k in range(steps):
            tau = torch.full((len(x), 1), k / steps)
            x += model(torch.cat([x, tau, goal], 1)) / steps
    return x.numpy()


def metrics(x, goal):
    modes = np.stack([np.c_[np.full(len(x), -.7), .6 * goal[:, 0]],
                      np.c_[np.full(len(x), .7), .6 * goal[:, 0]]], 1)
    return {'center_corridor_rate': float((np.abs(x[:, 0]) < .30).mean()),
            'positive_mode_fraction': float((x[:, 0] > 0).mean()),
            'nearest_mode_rmse': float(np.sqrt(((x[:, None] - modes)**2).sum(-1).min(1).mean())),
            'goal_coordinate_rmse': float(np.sqrt(((x[:, 1] - .6 * goal[:, 0])**2).mean()))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(32)
    train_g, train_a = data(4096, torch.Generator().manual_seed(320))
    test_g, test_a = data(2048, torch.Generator().manual_seed(321))
    model = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=.002)
    history = []
    for step in range(3000):
        ids = torch.randint(len(train_a), (256,))
        a, g = train_a[ids], train_g[ids]
        noise, tau = torch.randn_like(a), torch.rand(256, 1)
        x = (1 - tau) * noise + tau * a
        target_velocity = a - noise
        loss = (model(torch.cat([x, tau, g], 1)) - target_velocity).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            history.append([step, float(loss.detach())])
    design = np.c_[train_g.numpy(), np.ones(len(train_g))]
    coef = np.linalg.lstsq(design, train_a.numpy(), rcond=None)[0]
    mean_action = np.c_[test_g.numpy(), np.ones(len(test_g))] @ coef
    noise = torch.randn(2048, 2, generator=torch.Generator().manual_seed(322))
    report = {'verified_date': '2026-10-09', 'torch': torch.__version__, 'numpy': np.__version__,
              'train_pairs': 4096, 'test_pairs': 2048, 'steps': 3000,
              'parameters': sum(p.numel() for p in model.parameters()),
              'conditional_mean': metrics(mean_action, test_g.numpy()),
              'reference': metrics(test_a.numpy(), test_g.numpy()), 'integration': {}}
    saved = {'goal': test_g.numpy(), 'reference': test_a.numpy(), 'mean_action': mean_action,
             'history': np.array(history), 'initial_noise': noise.numpy()}
    for steps in [1, 4, 16, 64]:
        samples = sample(model, noise, test_g, steps)
        report['integration'][str(steps)] = metrics(samples, test_g.numpy())
        saved[f'flow_{steps}'] = samples
    np.savez_compressed(OUT / 'results.npz', **saved)
    (OUT / 'results.json').write_text(json.dumps(report, indent=2))
    torch.save(model.state_dict(), OUT / 'flow_model.pt')
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for a, values, title in zip(ax, [test_a.numpy(), mean_action, saved['flow_64']],
                                ['Held-out demonstrations', 'Squared-loss conditional mean', 'Learned flow: 64 Euler steps']):
        a.scatter(values[:, 0], values[:, 1], c=test_g[:, 0], cmap='coolwarm', s=5, alpha=.3)
        a.axvspan(-.3, .3, color='gray', alpha=.16)
        a.set(xlim=(-1.2, 1.2), ylim=(-1, 1), xlabel='Midpoint lateral y (m)', ylabel='Final x (m)', title=title)
    fig.savefig(OUT / 'action_modes.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for steps in [1, 4, 16, 64]:
        ax[0].hist(saved[f'flow_{steps}'][:, 0], bins=np.linspace(-1.2, 1.2, 60),
                   histtype='step', density=True, label=str(steps) + ' steps')
    ax[0].set(xlabel='Midpoint lateral y (m)', ylabel='Density', title='Same weights and initial noise; change only steps')
    ax[0].legend(fontsize=8)
    ax[1].plot(*np.array(history).T, color='#187f90')
    ax[1].set(xlabel='Training iteration', ylabel='Flow velocity MSE', title='Stochastic training loss, not rollout success')
    fig.savefig(OUT / 'flow_steps.png', dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
