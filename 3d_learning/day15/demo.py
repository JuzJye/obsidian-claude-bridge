"""D15: PointNet mechanism checks, CPU only; no training or benchmark claims."""
from pathlib import Path
import argparse
import json
import platform
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TinyPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Conv1d(3, 16, 1), nn.ReLU())
        self.encoder = nn.Sequential(nn.Conv1d(16, 32, 1), nn.ReLU(),
                                     nn.Conv1d(32, 64, 1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        self.segmenter = nn.Sequential(nn.Conv1d(80, 32, 1), nn.ReLU(), nn.Conv1d(32, 2, 1))

    def forward(self, points, valid=None):
        # points: B,N,3; Conv1d expects B,C,N. The kernel never mixes points.
        local = self.local(points.transpose(1, 2))
        encoded = self.encoder(local)
        if valid is not None:
            if not valid.any(dim=1).all():
                raise ValueError('Each sample must contain at least one valid point.')
            encoded = encoded.masked_fill(~valid[:, None, :], -torch.inf)
        global_feature = encoded.amax(dim=2)
        logits = self.classifier(global_feature)
        fused = torch.cat([local, global_feature[:, :, None].expand(-1, -1, points.shape[1])], 1)
        per_point = self.segmenter(fused).transpose(1, 2)
        return logits, per_point, encoded, global_feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--outlier', type=float, default=6.0)
    parser.add_argument('--out-dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / '_assets/3d_learning/day15')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    raw = rng.normal(size=(2, 128, 3)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
    raw *= np.array([1.0, .6, .3], dtype=np.float32)
    points = torch.from_numpy(raw)
    model = TinyPointNet().eval()
    permutation = torch.randperm(128)
    with torch.no_grad():
        logits, seg, encoded, global_feature = model(points)
        lp, sp, _, _ = model(points[:, permutation])
        invariant_error = (lp - logits).abs().max().item()
        equivariant_error = (sp - seg[:, permutation]).abs().max().item()
        wrong_alignment_error = (sp - seg).abs().max().item()
        # Same flatten classifier and same points; only their order changes.
        flatten = nn.Linear(128 * 3, 3)
        flatten_error = (flatten(points.flatten(1)) -
                         flatten(points[:, permutation].flatten(1))).abs().max().item()
        translated = points + torch.tensor([.3, -.2, .1])
        translation_error = (model(translated)[0] - logits).abs().max().item()
        critical = torch.unique(encoded[0].argmax(dim=1))
        kept = points[0:1, critical]
        critical_error = (model(kept)[3] - global_feature[0:1]).abs().max().item()
        outlier = torch.tensor([[[args.outlier, 0., 0.]]]).expand(2, -1, -1)
        contaminated = torch.cat([points, outlier], dim=1)
        outlier_error = (model(contaminated)[0] - logits).abs().max().item()
        # Invalid padding deliberately has large values. Mask belongs at the pool.
        padded = torch.cat([points, torch.full((2, 16, 3), 20.)], 1)
        valid = torch.arange(144)[None, :].expand(2, -1) < 128
        masked_error = (model(padded, valid)[0] - logits).abs().max().item()
        unmasked_error = (model(padded)[0] - logits).abs().max().item()
    labels = torch.tensor([0, 2])
    loss = nn.CrossEntropyLoss()(model(points)[0], labels)
    loss.backward()  # A gradient-flow check, not an optimizer step or training run.
    grad_norm = float(model.local[0].weight.grad.norm())
    assert invariant_error < 1e-6 and equivariant_error < 1e-6
    assert critical_error < 1e-6 and masked_error < 1e-6
    assert flatten_error > 1e-4 and unmasked_error > 1e-4 and grad_norm > 0
    report = {
        'seed': args.seed, 'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'device': 'cpu', 'training_steps': 0,
        'shapes': {'points': list(points.shape), 'encoded': list(encoded.shape),
                   'global': list(global_feature.shape), 'classification': list(logits.shape),
                   'segmentation': list(seg.shape)},
        'max_abs_logit_error_permutation': invariant_error,
        'max_abs_segmentation_error_aligned': equivariant_error,
        'max_abs_segmentation_error_wrong_alignment': wrong_alignment_error,
        'max_abs_flatten_logit_error_permutation': flatten_error,
        'max_abs_logit_error_translation': translation_error,
        'critical_point_count_of_128': len(critical), 'critical_only_global_error': critical_error,
        'outlier_x': args.outlier, 'outlier_logit_error': outlier_error,
        'invalid_padding_masked_error': masked_error, 'invalid_padding_unmasked_error': unmasked_error,
        'cross_entropy_random_model': loss.item(), 'first_layer_gradient_norm': grad_norm,
        'interpretation': 'Architecture and autograd checks only; weights are random, no trained accuracy.'}
    (args.out_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(args.out_dir / 'arrays.npz', points=raw, critical=critical.numpy(),
             local_features=encoded[0].detach().numpy(), global_feature=global_feature[0].numpy())
    fig = plt.figure(figsize=(11, 4.5), layout='constrained')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*raw[0].T, s=12, c='#9ba9b7', alpha=.5, label='All 128 points')
    ax.scatter(*raw[0, critical.numpy()].T, c='#e36b35', s=35, label=f'{len(critical)} channel winners')
    ax.set(xlabel='x', ylabel='y', zlabel='z', title='Critical set of this random encoder')
    ax.legend(loc='upper left', fontsize=8)
    ax = fig.add_subplot(122)
    ax.plot(global_feature[0].numpy(), label='All points', lw=3)
    ax.plot(model(kept)[3][0].detach().numpy(), '--', label='Only channel winners')
    ax.set(xlabel='Feature channel', ylabel='Max activation', title='Same 64-D pooled descriptor')
    ax.legend()
    fig.savefig(args.out_dir / 'critical-points.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    names = ['PointNet reorder', 'Seg aligned', 'Seg unaligned', 'Flatten reorder',
             'Translate', 'Outlier', 'Padding masked', 'Padding unmasked']
    values = [invariant_error, equivariant_error, wrong_alignment_error, flatten_error,
              translation_error, outlier_error, masked_error, unmasked_error]
    ax.bar(np.arange(8), values, color=['#288e87', '#288e87', '#dc8647', '#dc8647',
                                      '#dc8647', '#dc8647', '#288e87', '#dc8647'])
    ax.set_xticks(np.arange(8), names, rotation=23, ha='right')
    ax.set(ylabel='Maximum absolute output difference', title='Which changes are guaranteed to preserve the output?')
    for i, value in enumerate(values):
        ax.text(i, value + max(values) * .025, f'{value:.3g}', ha='center', fontsize=8)
    ax.set_ylim(0, max(values) * 1.18)
    fig.savefig(args.out_dir / 'invariance.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
