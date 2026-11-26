"""Train synthetic geometric retrieval with a hand-written cross-attention."""
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


def attention(q, k, v, valid):
    scores = q @ k.transpose(-1, -2) / np.sqrt(q.shape[-1])
    mask = valid[:, None, :]
    scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=-1) * mask
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights @ v, weights


def make_data(rng, n):
    k = rng.normal(size=(n, 6, 3))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    counts = rng.integers(3, 6, n)
    valid = np.arange(6)[None, :] < counts[:, None]
    target = np.array([rng.choice(c, 2, replace=False) for c in counts])
    q = k[np.arange(n)[:, None], target] + rng.normal(scale=.01, size=(n, 2, 3))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    values = rng.uniform(0, 1, size=(n, 6, 1))
    goal = values[np.arange(n)[:, None], target]
    # Padding intentionally resembles query 0 and has an invalid large value.
    k[~valid] = np.broadcast_to(q[:, :1, :], k.shape)[~valid]
    values[~valid] = 5
    return [torch.tensor(a, dtype=torch.bool if a.dtype == bool else torch.long if a.dtype.kind in 'iu' else torch.float32) for a in (q, k, values, valid, target, goal)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=66)
    ap.add_argument('--ignore-mask', action='store_true')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / '_assets/3d_learning/day66')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val, test = [make_data(rng, n) for n in (512, 128, 128)]
    wq, wk = nn.Linear(3, 16, bias=False), nn.Linear(3, 16, bias=False)
    opt = torch.optim.Adam(list(wq.parameters()) + list(wk.parameters()), lr=.02)
    curve, best, state = [], float('inf'), None
    start = time.perf_counter()
    for step in range(301):
        q, k, v, mask, target, goal = train
        out, a = attention(wq(q), wk(k), v, mask)
        loss = -a.gather(-1, target[..., None]).clamp_min(1e-10).log().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            q, k, v, mask, target, goal = val
            out, a = attention(wq(q), wk(k), v, mask)
            vl = -a.gather(-1, target[..., None]).clamp_min(1e-10).log().mean()
        curve.append([step, loss.item(), vl.item()])
        if vl.item() < best:
            best, best_step = vl.item(), step
            state = (wq.weight.detach().clone(), wk.weight.detach().clone())
    train_seconds = time.perf_counter() - start
    with torch.no_grad():
        wq.weight.copy_(state[0]); wk.weight.copy_(state[1])
    q, k, v, mask, target, goal = test
    records, weights = {}, {}
    with torch.no_grad():
        for name, m in [('masked', mask), ('ignored', torch.ones_like(mask))]:
            start = time.perf_counter(); out, a = attention(wq(q), wk(k), v, m)
            records[name] = {'retrieval_correct': int((a.argmax(-1) == target).sum()), 'query_count': int(target.numel()),
                             'value_mae': float((out - goal).abs().mean()), 'padded_mass': float((a * ~mask[:, None, :]).sum(-1).mean()),
                             'batch_latency_ms': (time.perf_counter() - start) * 1000}
            weights[name] = a.numpy()
        zero_out, zero_a = attention(wq(q[:1]), wk(k[:1]), v[:1], torch.zeros_like(mask[:1]))
        perm = torch.tensor([5, 2, 0, 4, 1, 3])
        ref, a = attention(wq(q), wk(k), v, mask)
        equiv, _ = attention(wq(q), wk(k[:, perm]), v[:, perm], mask[:, perm])
    checks = {'all_empty_output_max_abs': float(zero_out.abs().max()), 'all_empty_weights_sum': float(zero_a.sum()),
              'key_value_joint_permutation_max_abs': float((ref - equiv).abs().max())}
    assert checks['all_empty_output_max_abs'] == 0 and checks['key_value_joint_permutation_max_abs'] < 1e-5
    result = {'seed': args.seed, 'selected': 'ignored' if args.ignore_mask else 'masked', 'results': records, 'checks': checks,
              'best_step': best_step, 'train_steps': 301, 'train_seconds': train_seconds,
              'samples': {'train': 512, 'validation': 128, 'test': 128}, 'shapes': {'query': [128, 2, 3], 'key': [128, 6, 3], 'value': [128, 6, 1], 'attention': [128, 2, 6]},
              'supervision': 'known synthetic geometric correspondence index; no language or semantic training',
              'units': 'unit-sphere directions and dimensionless contact proxy, padding value 5 invalid', 'robot_trials': 0,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'matplotlib': matplotlib.__version__, 'device': 'cpu'}, 'project_acceptance': 'pending'}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, name in zip(axes, weights):
        im = ax.imshow(weights[name][0], vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(6), [f'{i}' + (' PAD' if not mask[0, i] else '') for i in range(6)])
        ax.set(xlabel='Tactile key index', ylabel='Visual query index', yticks=[0, 1], title=f'{name}; target={target[0].tolist()}')
        fig.colorbar(im, ax=ax, label='Attention weight')
    fig.tight_layout(); fig.savefig(args.out / 'attention.png', dpi=150); plt.close(fig)
    c = np.array(curve); fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(c[:, 0], c[:, 1], label='train'); axes[0].plot(c[:, 0], c[:, 2], label='validation')
    axes[0].set(xlabel='Optimization step', ylabel='Correspondence NLL', title='Supervised synthetic retrieval'); axes[0].legend()
    axes[1].bar(list(records), [r['value_mae'] for r in records.values()], color=['#228899', '#dd7755'])
    axes[1].set(ylabel='Value MAE', title='Only inference mask changes')
    fig.tight_layout(); fig.savefig(args.out / 'retrieval.png', dpi=150); plt.close(fig)
    print(json.dumps({'results': records, 'checks': checks, 'best_step': best_step}, indent=2))


if __name__ == '__main__':
    main()
