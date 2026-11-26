"""CPU synthetic joint encoder; normalized xyz proxy is not a 6D pose model."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=65)
    ap.add_argument('--shuffle-test', action='store_true')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / '_assets/3d_learning/day65')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    groups = np.repeat(np.arange(120), 8)
    z = rng.normal(size=(120, 3))[groups] * .5 + rng.normal(size=(960, 3)) * .7
    v = np.column_stack([z[:, 0], z[:, 1], z[:, 0] * z[:, 1], np.sin(z[:, 0]), np.cos(z[:, 1]), z[:, 2] * .05])
    t = np.column_stack([z[:, 2], np.tanh(z[:, 2]), z[:, 1] * .2, z[:, 0] * .1, z[:, 2] ** 2, np.ones(960)])
    v += rng.normal(scale=.12, size=v.shape)
    t += rng.normal(scale=.12, size=t.shape)
    y = (z[:, 0] + z[:, 2] > .2).astype(np.float32)
    split = {'train': groups < 80, 'val': (groups >= 80) & (groups < 100), 'test': groups >= 100}
    norm = []
    for a in (v, t):
        mu, sd = a[split['train']].mean(0), a[split['train']].std(0)
        sd = np.maximum(sd, 1e-6)
        norm.append({'mean': mu.tolist(), 'std': sd.tolist()})
        a -= mu
        a /= sd
    v, t, z, y = [torch.tensor(a, dtype=torch.float32) for a in (v, t, z, y)]

    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 8))
            self.tactile = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 8))
            self.fuse = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
            self.xyz = nn.Linear(16, 3)
            self.contact = nn.Linear(16, 1)
        def forward(self, a, b):
            h = self.fuse(torch.cat([self.visual(a), self.tactile(b)], dim=-1))
            return self.xyz(h), self.contact(h).squeeze(-1)

    net = Joint()
    opt = torch.optim.Adam(net.parameters(), lr=.01)
    best, best_state, curve = float('inf'), None, []
    start = time.perf_counter()
    for epoch in range(401):
        net.train()
        xyz, logit = net(v[split['train']], t[split['train']])
        loss = ((xyz - z[split['train']]) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(logit, y[split['train']])
        opt.zero_grad()
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            xyz, logit = net(v[split['val']], t[split['val']])
            val = ((xyz - z[split['val']]) ** 2).mean() + nn.functional.binary_cross_entropy_with_logits(logit, y[split['val']])
        curve.append([epoch, loss.item(), val.item()])
        if val.item() < best:
            best, best_epoch = val.item(), epoch
            best_state = {k: a.detach().clone() for k, a in net.state_dict().items()}
    train_seconds = time.perf_counter() - start
    net.load_state_dict(best_state)
    net.eval()
    vi, ti, zi, yi = [a[split['test']] for a in (v, t, z, y)]
    perm = torch.tensor(np.random.default_rng(args.seed + 1).permutation(len(ti)))
    outputs, rows = {}, {}
    for name, tactile in [('aligned', ti), ('shuffled', ti[perm])]:
        start = time.perf_counter()
        with torch.no_grad():
            pred, logit = net(vi, tactile)
        latency = (time.perf_counter() - start) * 1000
        err = torch.linalg.vector_norm((pred - zi) * 30, dim=-1).numpy()
        outputs[name] = pred.numpy()
        rows[name] = {'xyz_rmse_mm': float(torch.sqrt(((pred - zi) * 30).square().mean())),
                      'contact_correct': int(((logit >= 0) == yi.bool()).sum()), 'contact_total': len(yi),
                      'contact_accuracy': float(((logit >= 0) == yi.bool()).float().mean()),
                      'batch_latency_ms': latency, 'mean_position_l2_mm': float(err.mean())}
    result = {'seed': args.seed, 'data': 'synthetic complementary feature vectors, no image encoder or real contact',
              'units': 'latent xyz scaled by 30 mm; binary contact is synthetic rule',
              'split_groups': {'train': [0, 79], 'val': [80, 99], 'test': [100, 119]},
              'split_samples': {k: int(a.sum()) for k, a in split.items()},
              'best_epoch': best_epoch, 'train_seconds': train_seconds, 'train_steps': 401,
              'selected': 'shuffled' if args.shuffle_test else 'aligned', 'evaluation': rows,
              'normalization_fit': 'training only', 'normalization': norm,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'matplotlib': matplotlib.__version__, 'device': 'cpu'},
              'robot_trials': 0, 'project_acceptance': 'pending'}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    torch.save({'weights': best_state, 'normalization': norm}, args.out / 'joint_encoder.pt')
    c = np.asarray(curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c[:, 0], c[:, 1], label='train')
    ax.plot(c[:, 0], c[:, 2], label='validation')
    ax.axvline(best_epoch, color='gray', ls='--', label='selected epoch')
    ax.set(xlabel='Optimization step', ylabel='MSE + BCE', title='Joint encoder: grouped holdout')
    ax.legend(); fig.tight_layout(); fig.savefig(args.out / 'training.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in outputs:
        axes[0].scatter(zi.numpy()[:, 2] * 30, outputs[name][:, 2] * 30, s=12, alpha=.5, label=name)
    axes[0].plot([-70, 70], [-70, 70], 'k--'); axes[0].set(xlabel='True z (mm)', ylabel='Predicted z (mm)', title='Same weights, changed pairing'); axes[0].legend()
    axes[1].bar(list(rows), [rows[n]['contact_accuracy'] for n in rows], color=['#228899', '#dd7755'])
    axes[1].set(ylim=(0, 1), ylabel='Contact accuracy', title='160 held-out samples per condition')
    fig.tight_layout(); fig.savefig(args.out / 'pairing.png', dpi=150); plt.close(fig)
    print(json.dumps({k: result[k] for k in ['best_epoch', 'selected', 'evaluation']}, indent=2))


if __name__ == '__main__':
    main()
