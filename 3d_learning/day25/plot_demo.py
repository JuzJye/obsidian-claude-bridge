"""Render Day25 figures from already computed arrays; no retraining."""
from pathlib import Path
import argparse
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'d25-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day25')
    args = p.parse_args()
    a = np.load(args.input/'arrays.npz')
    r = json.loads((args.input/'results.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    for i, key in enumerate(['hash_truth', 'hash_16', 'hash_512']):
        title = 'Analytic target' if i == 0 else f'Hash cap {16 if i == 1 else 512}'
        ax[0, i].imshow(a[key], origin='lower', extent=[0, 1, 0, 1], vmin=.1, vmax=.9, cmap='viridis')
        ax[0, i].set(title=title, xlabel='x (normalized)', ylabel='y (normalized)')
    for cap in [16, 512]:
        loss = a[f'loss_{cap}']
        ax[1, 0].semilogy(loss[:, 0], loss[:, 1], label=f'cap {cap}')
    ax[1, 0].set(title='Training batch MSE', xlabel='Optimizer update', ylabel='MSE')
    ax[1, 0].legend()
    for i, cap in enumerate([16, 512], start=1):
        im = ax[1, i].imshow(abs(a[f'hash_{cap}']-a['hash_truth']), origin='lower',
                             extent=[0, 1, 0, 1], cmap='magma', vmin=0, vmax=.3)
        ax[1, i].set(title=f'Test absolute error; RMSE={r["hash"][str(cap)]["test_rmse"]:.4f}', xlabel='x', ylabel='y')
    fig.colorbar(im, ax=ax[1, 1:], shrink=.8, label='Absolute signal error')
    fig.suptitle('2D coordinate fitting: mechanism demo, not a NeRF benchmark')
    fig.savefig(args.input/'hash-capacity.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax[0].imshow(a['splat_correct'])
    ax[1].imshow(a['splat_wrong_order'])
    im = ax[2].imshow(abs(a['splat_correct']-a['splat_wrong_order']).mean(-1), cmap='magma', vmin=0, vmax=.3)
    for axis, title in zip(ax, ['Front to back: red then blue', 'Wrong order: blue then red', 'Mean RGB absolute difference']):
        axis.set(title=title, xlabel='u (px)', ylabel='v (px)')
    fig.colorbar(im, ax=ax[2], shrink=.8)
    fig.suptitle('Same projected Gaussians; compositing order is the only change')
    fig.savefig(args.input/'gaussian-order.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    steps = np.arange(1, 11)
    for label in ['learned', 'biased']:
        ax[0].plot(steps, 1000*a[f'id_{label}_free_rmse'][:, 0], marker='o', label=f'{label}: free')
        ax[0].plot(steps, 1000*a[f'id_{label}_teacher_rmse'][:, 0], linestyle='--', label=f'{label}: reset to truth')
    ax[0].set(title='In-distribution commands', xlabel='Horizon step (dt=0.1s)', ylabel='Position RMSE (mm)')
    ax[0].legend(fontsize=8)
    for label in ['learned', 'known_limit']:
        ax[1].plot(steps, 1000*a[f'ood_{label}_free_rmse'][:, 0], marker='o', label=f'{label}: free')
        ax[1].plot(steps, 1000*a[f'ood_{label}_teacher_rmse'][:, 0], linestyle='--', label=f'{label}: reset to truth')
    ax[1].set(title='OOD commands: unseen saturation', xlabel='Horizon step', ylabel='Position RMSE (mm)')
    ax[1].legend(fontsize=8)
    idx = int(np.argmax(abs(a['ood_actions'][:, 0])))
    for key, name in [('ood_truth', 'True trajectory'), ('ood_learned_rollout', 'Learned, free'),
                       ('ood_known_limit_rollout', 'Known limit, free')]:
        ax[2].plot(np.arange(11)*.1, a[key][idx, :, 0], marker='.', label=name)
    ax[2].set(title=f'Largest command: a={a["ood_actions"][idx,0]:.2f} m/s²', xlabel='Time (s)', ylabel='Position (m)')
    ax[2].legend(fontsize=8)
    fig.suptitle('Fixed commanded action sequences: prediction evaluation, not closed-loop control')
    fig.savefig(args.input/'rollout-errors.png', dpi=160)
    plt.close(fig)
    print('Wrote hash-capacity.png, gaussian-order.png, rollout-errors.png')


if __name__ == '__main__':
    main()
