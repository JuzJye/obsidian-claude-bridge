"""Contact-event learning plus a one-step offline support counterexample."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/day33-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/day33-cache')
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / '_assets/3d_learning/day33'


def scores(p, y, threshold):
    positive = p >= threshold
    return {'brier': float(np.mean((p - y)**2)),
            'false_negative_rate': float(np.mean(~positive[y == 1])),
            'false_positive_rate': float(np.mean(positive[y == 0])),
            'accuracy': float(np.mean(positive == y))}


def fit(features, y, train):
    mu, std = features[train].mean(0), features[train].std(0)
    x = torch.tensor((features - mu) / std, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)[:, None]
    model = nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=.04)
    for _ in range(1600):
        logits = model(x[train])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target[train])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).numpy()[:, 0]
    return prob, mu, std, model.weight.detach().numpy(), model.bias.detach().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(33); torch.set_num_threads(2)
    rng = np.random.default_rng(33)
    # Each independent acquisition session has a fixed unknown surface offset.
    sessions = np.repeat(np.arange(150), 20)
    split = np.where(sessions < 100, 0, np.where(sessions < 125, 1, 2))
    offset = rng.normal(0, .002, 150)
    gap = rng.uniform(-.005, .04, 3000)
    velocity = rng.uniform(0., .20, 3000)
    acceleration = rng.uniform(-.8, .8, 3000)
    horizon = .1
    # First-crossing under constant acceleration: include interior extremum.
    t_vertex = np.clip(-velocity / np.where(np.abs(acceleration) > 1e-12, acceleration, 1e-12), 0, horizon)
    advance_end = velocity * horizon + .5 * acceleration * horizon**2
    advance_vertex = velocity * t_vertex + .5 * acceleration * t_vertex**2
    advance_max = np.maximum.reduce([np.zeros(3000), advance_end, advance_vertex])
    margin = gap + offset[sessions] - advance_max
    labels = (margin <= 0).astype(float)
    features = np.c_[gap, velocity, acceleration]
    full, mu, std, weight, bias = fit(features, labels, split == 0)
    gap_only, *_ = fit(features[:, :1], labels, split == 0)
    test = split == 2
    report = {'verified_date': '2026-10-09', 'numpy': np.__version__, 'torch': torch.__version__,
              'horizon_s': horizon, 'sessions': [100, 25, 25], 'probes_per_session': 20,
              'surface_offset_std_m': .002, 'test_prevalence': float(labels[test].mean()),
              'full_threshold_05': scores(full[test], labels[test], .5),
              'full_threshold_02': scores(full[test], labels[test], .2),
              'gap_only_threshold_05': scores(gap_only[test], labels[test], .5)}
    action_data = np.linspace(-.6, .6, 61)
    reward_data = action_data.copy()  # all observed trials terminate in one step
    coef = np.polyfit(action_data, reward_data, 1)
    grid = np.linspace(-1.5, 1.5, 301)
    q = np.polyval(coef, grid)
    true_reward = np.where(np.abs(grid) <= .8, grid, -5.)
    distance = np.maximum(np.abs(grid) - .6, 0.)
    pessimistic = q - 2. * distance
    report['offline_support'] = {}
    for name, value in [('unrestricted', q), ('distance_penalty', pessimistic)]:
        k = int(np.argmax(value))
        report['offline_support'][name] = {'action': float(grid[k]), 'model_score': float(value[k]),
                                           'true_reward': float(true_reward[k])}
    report['offline_support']['train_mse'] = float(np.mean((np.polyval(coef, action_data)-reward_data)**2))
    np.savez_compressed(OUT / 'results.npz', features=features, labels=labels, split=split,
                        session_ids=sessions, full_probability=full, gap_probability=gap_only,
                        train_mean=mu, train_std=std, weight=weight, bias=bias,
                        grid=grid, q=q, true_reward=true_reward, penalized=pessimistic)
    (OUT / 'results.json').write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    points = ax[0].scatter(gap[test]*1000, velocity[test]*1000, c=full[test], cmap='viridis', s=18)
    ax[0].set(xlabel='Observed gap (mm)', ylabel='Closing velocity (mm/s)', title='Held-out sessions: predicted contact probability')
    fig.colorbar(points, ax=ax[0], label='P(contact within 0.1 s)')
    bins = np.linspace(0, 1, 9)
    for pred, name in [(full, 'gap + velocity + acceleration'), (gap_only, 'gap only')]:
        bx, by = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            keep = test & (pred >= lo) & (pred < hi + (1e-9 if hi == 1 else 0))
            if keep.sum() >= 5:
                bx.append(pred[keep].mean()); by.append(labels[keep].mean())
        ax[1].plot(bx, by, 'o-', label=name)
    ax[1].plot([0, 1], [0, 1], 'k--', alpha=.5)
    ax[1].set(xlabel='Mean predicted probability', ylabel='Observed contact fraction', title='Reliability bins (at least 5 samples)')
    ax[1].legend(fontsize=8)
    fig.savefig(OUT / 'contact_probability.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.axvspan(-.6, .6, color='gray', alpha=.18, label='Dataset action range')
    ax.plot(grid, true_reward, label='True terminal reward')
    ax.plot(grid, q, '--', label='Linear fitted Q: perfect on dataset')
    ax.plot(grid, pessimistic, label='Q minus 2 x distance outside support')
    ax.set(xlabel='Normalized action', ylabel='Reward / model score', title='Offline data do not identify unobserved consequences')
    ax.legend(fontsize=8)
    fig.savefig(OUT / 'offline_support.png', dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
