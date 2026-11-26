"""D51: synthetic instance ROI -> camera-frame cloud; no learned model or hardware."""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_scene(seed):
    rng = np.random.default_rng(seed)
    h, w = 128, 192
    v, u = np.mgrid[:h, :w]
    body = ((u - 84) / 26) ** 2 + ((v - 65) / 34) ** 2 <= 1
    outer = ((u - 110) / 17) ** 2 + ((v - 63) / 21) ** 2 <= 1
    inner = ((u - 110) / 10) ** 2 + ((v - 63) / 14) ** 2 < 1
    amodal = body | (outer & ~inner)
    neighbor = ((u - 124) / 17) ** 2 + ((v - 92) / 19) ** 2 <= 1
    occluder = (u > 60) & (u < 76) & (v > 62) & (v < 110)
    labels = np.zeros((h, w), np.uint8)
    labels[amodal] = 1
    labels[neighbor] = 2
    labels[occluder] = 3
    target = labels == 1  # visible target, evaluation only
    depth = np.full((h, w), 1.1, dtype=float)
    depth[target] = .72 + .012 * ((u[target] - 84) / 26) ** 2
    depth[labels == 2] = .73  # similar depth: gating cannot separate identity
    depth[labels == 3] = .55
    depth += rng.normal(0, .001, depth.shape)
    depth[rng.random(depth.shape) < .025] = 0
    palette = np.array([[.14, .17, .22], [.92, .56, .20], [.23, .67, .53], [.50, .48, .76]])
    rgb = palette[labels]
    # Controlled mock network output, explicitly constructed from labels.
    probability = cv2.GaussianBlur(target.astype(np.float32), (9, 9), 1.5)
    probability[neighbor] = np.maximum(probability[neighbor], .68)
    probability = np.clip(probability + rng.normal(0, .025, depth.shape), 0, 1)
    k = np.array([[170., 0, 96], [0, 170., 64], [0, 0, 1.]])
    return rgb, depth, labels, probability, k


def shift_mask(probability, shift):
    transform = np.array([[1., 0, shift], [0, 1., 0]])
    return cv2.warpAffine(probability, transform,
                          (probability.shape[1], probability.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def extract(depth, mask, k):
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    return np.column_stack(((u-k[0, 2])*z/k[0, 0],
                            (v-k[1, 2])*z/k[1, 1], z))


def evaluate(depth, labels, probability, k, shift):
    shifted = shift_mask(probability, shift)
    v, u = np.mgrid[:depth.shape[0], :depth.shape[1]]
    roi = (u >= 54) & (u < 144) & (v >= 28) & (v < 113)
    valid = np.isfinite(depth) & (depth > 0)
    predicted = (shifted >= .5) & roi
    seeds = roi & valid & (shifted > .90)
    center_z = float(np.median(depth[seeds])) if np.any(seeds) else np.nan
    gated = predicted & (np.abs(depth - center_z) < .035)
    reference = extract(depth, labels == 1, k).mean(axis=0)
    masks = {'box': roi, 'instance_mask': predicted, 'mask_depth_gate': gated}
    reports, clouds = {}, {}
    denominator = int(np.sum((labels == 1) & valid))
    for name, mask in masks.items():
        accepted = mask & valid
        count = int(np.sum(accepted))
        correct = int(np.sum(accepted & (labels == 1)))
        cloud = extract(depth, accepted, k)
        clouds[name] = cloud
        reports[name] = dict(points=count, target_points=correct,
            retention=correct/denominator,
            contamination=(count-correct)/count if count else None,
            visible_centroid_error_mm=float(np.linalg.norm(cloud.mean(0)-reference)*1000)
            if count else None)
    return reports, clouds, shifted, reference, center_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift-px', type=float, default=0)
    parser.add_argument('--seed', type=int, default=51)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day51')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rgb, depth, labels, probability, k = make_scene(args.seed)
    reports, clouds, shifted, reference, center_z = evaluate(depth, labels, probability, k, args.shift_px)
    elapsed_ms = (time.perf_counter()-started)*1000
    result = dict(seed=args.seed, shift_px=args.shift_px, depth_gate_center_m=center_z,
        metrics=reports, frame='camera_optical: x right, y down, z forward',
        length_unit='m', image_shape=list(depth.shape), data='procedural mock RGB-D and mask probabilities',
        timestamp_s=0.0, pairing='same synthetic instant', physical_robot_trials=0,
        training_samples=0, synthetic_generation_and_geometry_ms=elapsed_ms,
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     opencv=cv2.__version__, matplotlib=matplotlib.__version__))
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.out/'results.npz', rgb=rgb, depth=depth, labels=labels,
        probability=shifted, K=k, selected_cloud=clouds['mask_depth_gate'])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title('Synthetic RGB: orange target / green neighbor')
    axes[0, 0].add_patch(plt.Rectangle((54, 28), 90, 85, fill=False, color='white', lw=1.5))
    im = axes[0, 1].imshow(np.ma.masked_equal(depth, 0), vmin=.5, vmax=1.12, cmap='viridis')
    axes[0, 1].set_title('Aligned axial depth Z [m]; holes masked')
    fig.colorbar(im, ax=axes[0, 1], shrink=.8)
    axes[1, 0].imshow(labels, vmin=0, vmax=3, cmap='tab10'); axes[1, 0].set_title('Evaluation labels: visible instances')
    im = axes[1, 1].imshow(shifted, vmin=0, vmax=1, cmap='magma')
    axes[1, 1].contour(shifted >= .5, levels=[.5], colors='cyan', linewidths=.7)
    axes[1, 1].set_title(f'Mock probability; horizontal shift={args.shift_px:g} px')
    fig.colorbar(im, ax=axes[1, 1], shrink=.8)
    for ax in axes.flat:
        ax.set_xlabel('u [px]'); ax.set_ylabel('v [px]')
    fig.tight_layout(); fig.savefig(args.out/'roi_scene.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, cloud in clouds.items():
        axes[0].scatter(cloud[::4, 0], cloud[::4, 2], s=5, alpha=.4, label=name)
        axes[0].plot(cloud[:, 0].mean(), cloud[:, 2].mean(), 'x', markersize=10)
    axes[0].plot(reference[0], reference[2], 'k*', ms=12, label='visible GT centroid')
    axes[0].set(xlabel='camera X [m]', ylabel='camera Z [m]', title='Selected cloud and centroid (not object origin)')
    axes[0].legend(fontsize=7)
    shifts = np.arange(0, 13, 2)
    for name in clouds:
        errors = [evaluate(depth, labels, probability, k, float(s))[0][name]['visible_centroid_error_mm'] for s in shifts]
        axes[1].plot(shifts, errors, 'o-', label=name)
    axes[1].set(xlabel='mask-depth horizontal mismatch [px]', ylabel='visible-centroid error [mm]', title='One variable: keep RGB-D fixed, shift mask')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.out/'roi_cloud_error.png', dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
