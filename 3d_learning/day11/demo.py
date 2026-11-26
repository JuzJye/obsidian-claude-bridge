"""Day11: CPU stereo costs, depth sensitivity, gradient learning, split leakage.
Synthetic diagnostics only: not MC-CNN/GC-Net training or a real-scene benchmark.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import cv2
import numpy as np


def costs(left, right, disparities=24, window=7):
    h, w = left.shape
    volume = np.full((disparities, h, w), 1e5, np.float32)
    for d in range(disparities):
        c = np.full((h, w), 255.0, np.float32)
        c[:, d:] = np.abs(left[:, d:] - right[:, :w-d])
        volume[d] = cv2.boxFilter(c, -1, (window, window), normalize=True)
        volume[d, :, :d + window // 2] = 1e5
    return volume


def scores(disp, truth, mask):
    valid = mask & np.isfinite(disp) & (disp > 0)
    error = np.abs(disp[valid] - truth)
    return dict(coverage=float(valid.sum() / mask.sum()), epe_px=float(error.mean()),
                bad1_fraction=float(np.mean(error > 1)), n=int(valid.sum()))


def train_affine(seed):
    rng = np.random.default_rng(seed)
    measured = rng.uniform(0.5, 2.5, 160)
    truth = 1.1 * measured + 0.03 + rng.normal(0, .01, 160)
    train, valid, test = np.arange(100), np.arange(100, 130), np.arange(130, 160)
    # Standardization is fitted on train only; validate selects no hyperparameter here.
    mu, sd = measured[train].mean(), measured[train].std()
    x = np.column_stack(((measured - mu) / sd, np.ones(160)))
    theta = np.zeros(2)
    history = []
    for epoch in range(60):
        for batch in np.array_split(rng.permutation(train), 5):
            residual = x[batch] @ theta - truth[batch]
            grad = 2 * x[batch].T @ residual / len(batch)
            theta -= .08 * grad
        history.append([np.mean((x[train] @ theta - truth[train])**2),
                        np.mean((x[valid] @ theta - truth[valid])**2)])
    test_rmse = np.sqrt(np.mean((x[test] @ theta - truth[test])**2))
    physical = [theta[0] / sd, theta[1] - theta[0] * mu / sd]
    return dict(coefficients=physical, test_rmse_m=float(test_rmse),
                split_sizes=[100,30,30], batch_size=20, epochs=60, updates=300), np.array(history)


def leakage(seed):
    rng = np.random.default_rng(seed)
    groups, frames = 40, 10
    # A scene fingerprint plus an arbitrary scene-specific label: no transferable rule.
    identity = np.repeat(np.eye(groups), frames, axis=0)
    x = identity + rng.normal(0, .002, identity.shape)
    labels = np.repeat(rng.normal(0, .3, groups), frames)
    scene = np.repeat(np.arange(groups), frames)
    shuffled = rng.permutation(len(labels))
    splits = dict(random_frames=(shuffled[:280], shuffled[280:]),
                  grouped_scenes=(np.where(scene < 28)[0], np.where(scene >= 28)[0]))
    result = {}
    for name, (tr, te) in splits.items():
        distances = ((x[te, None] - x[None, tr])**2).sum(2)
        predicted = labels[tr[np.argmin(distances, axis=1)]]
        result[name] = dict(rmse=float(np.sqrt(np.mean((predicted-labels[te])**2))),
                            shared_scenes=int(len(set(scene[tr]) & set(scene[te]))))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day11')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    h, w, truth = 80, 160, 8
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (3, 3), .6)
    texture[25:55, 70:105] = 127.0
    right_clean = np.empty_like(texture)
    right_clean[:, :-truth] = texture[:, truth:]
    right_clean[:, -truth:] = rng.uniform(0, 255, (h, truth))
    left = np.clip(texture + rng.normal(0, 3, texture.shape), 0, 255).astype(np.float32)
    noise = rng.normal(0, 3, texture.shape)
    mask = np.zeros((h, w), bool)
    mask[12:-12, 32:-12] = True  # Fixed common support for every comparison.
    textured = mask.copy(); textured[20:60,65:110] = False
    flat = mask.copy(); flat[:] = False; flat[33:47,80:95] = True
    records, arrays = {}, dict(left=left, mask=mask)
    started = time.perf_counter()
    for offset in (0, 30):  # Single changed condition: right-image brightness offset.
        right = np.clip(right_clean + noise + offset, 0, 255).astype(np.float32)
        for window in (1, 7):
            volume = costs(left, right, window=window)
            disp = np.argmin(volume, axis=0).astype(np.float32)
            key = f'sad{window}_offset{offset}'
            records[key] = dict(all=scores(disp, truth, mask), textured=scores(disp, truth, textured), flat=scores(disp, truth, flat))
            arrays[key] = disp
            if offset == 0 and window == 7:
                arrays['cost_textured'] = volume[:, 40, 45]
                arrays['cost_flat'] = volume[:, 40, 88]
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=5,
                P1=8*25, P2=32*25, uniquenessRatio=5, speckleWindowSize=0,
                disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(left.astype(np.uint8), right.astype(np.uint8)).astype(np.float32)/16
        records[f'sgbm_offset{offset}'] = dict(all=scores(disp, truth, mask), textured=scores(disp, truth, textured), flat=scores(disp, truth, flat))
        arrays[f'sgbm_offset{offset}'] = disp
        arrays[f'right_offset{offset}'] = right
    stereo_ms = 1000*(time.perf_counter()-started)
    learning, arrays['loss_history'] = train_affine(args.seed)
    leakage_result = leakage(args.seed)
    z = np.array([.5,1,2,4]); f, baseline, sigma_d = 600., .06, .25
    sigma_z = z*z/(f*baseline)*sigma_d
    arrays['depth'] = z; arrays['sigma_z'] = sigma_z
    bimodal = np.array([0,0,.5,0,0,0,0,0,.5,0], float)
    report = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__),
        seed=args.seed, truth_disparity_px=truth, stereo=records, stereo_runtime_ms=stereo_ms,
        depth_sigma=dict(z_m=z.tolist(), sigma_d_px=sigma_d, sigma_z_m=sigma_z.tolist()),
        affine_learning=learning, leakage=leakage_result,
        soft_argmin_bimodal=dict(modes=[2,8], expectation=float(np.arange(10)@bimodal)),
        scope='synthetic rectified plane; no learned stereo model training; affine training only')
    assert records['sad7_offset0']['textured']['epe_px'] < .1
    assert learning['test_rmse_m'] < .03
    assert leakage_result['random_frames']['shared_scenes'] > 0
    assert leakage_result['grouped_scenes']['shared_scenes'] == 0
    (args.out_dir/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(args.out_dir/'arrays.npz', **arrays)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
