"""D13 original-plan experiments: matching, soft argmin, and camera calibration.

CPU synthetic data only; requires NumPy, OpenCV, and Matplotlib. No training.
"""
import json
import platform
from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def soft_argmin(cost, temperature=1.):
    logits = -np.asarray(cost, float) / temperature
    weight = np.exp(logits - logits.max())
    weight /= weight.sum()
    return float(weight @ np.arange(len(weight))), weight


def main():
    out = Path(__file__).resolve().parents[2] / '_assets/3d_learning/day13'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1313)
    begin = time.perf_counter()
    h, w, disparity = 48, 128, 6
    left = rng.uniform(0, 1, (h, w)).astype(np.float32)
    right = rng.uniform(0, 1, (h, w)).astype(np.float32)
    right[:, :-disparity] = left[:, disparity:]
    right += rng.normal(0, .015, right.shape).astype(np.float32)
    volume = np.full((13, h, w), np.inf, np.float32)
    for d in range(len(volume)):
        raw = np.ones_like(left)
        raw[:, d:] = abs(left[:, d:] - right[:, :w-d])
        volume[d] = cv2.boxFilter(raw, -1, (5, 5), normalize=True)
        volume[d, :, :d+2] = np.inf
    hard = volume.argmin(axis=0)
    roi = (slice(2, -2), slice(14, -2))
    accuracy = float(np.mean(hard[roi] == disparity))
    assert accuracy > .99
    curve = volume[:, 20, 60].astype(float)
    soft_cases = []
    for temp in [.02, .1, 1.]:
        estimated, probs = soft_argmin(curve, temp)
        soft_cases.append({'temperature': temp, 'disparity_px': estimated,
                           'probabilities': probs.tolist()})
    two_peak = np.full(13, 100., float)
    two_peak[[3, 9]] = 0.
    mean, prob = soft_argmin(two_peak)
    assert np.isclose(mean, 6.) and prob[6] < 1e-30
    # Differentiate the teaching soft argmin, not a trained GC-Net.
    temp = .1
    mu, p = soft_argmin(curve, temp)
    derivative = -p * (np.arange(len(p)) - mu) / temp
    eps = 1e-6
    numeric = np.array([(soft_argmin(curve + np.eye(13)[k]*eps, temp)[0]
                         - soft_argmin(curve - np.eye(13)[k]*eps, temp)[0])
                        / (2*eps) for k in range(13)])
    assert np.max(abs(derivative-numeric)) < 1e-6

    board = np.zeros((54, 3), np.float32)
    board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * .025
    K = np.array([[720., 0, 320], [0, 710, 240], [0, 0, 1]])
    distortion = np.array([-.18, .04, 0., 0., 0.])
    objects, images, true_t = [], [], []
    for _ in range(20):
        rv = rng.uniform([-.5, -.45, -.15], [.5, .45, .15])
        tv = rng.uniform([-.15, -.10, .65], [.0, .03, .95])
        uv, _ = cv2.projectPoints(board, rv, tv, K, distortion)
        assert np.all((uv[..., 0] > 0) & (uv[..., 0] < 640))
        assert np.all((uv[..., 1] > 0) & (uv[..., 1] < 480))
        uv += rng.normal(0, .12, uv.shape).astype(np.float32)
        objects.append(board.copy())
        images.append(uv.astype(np.float32))
        true_t.append(tv)
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_LU
    fits = []
    for scale in [1., 1.05]:
        rms, khat, dhat, rhats, thats = cv2.calibrateCamera(
            [x*scale for x in objects], images, (640, 480), None, None,
            flags=flags)
        fits.append((rms, khat, dhat, rhats, thats))
    rms, khat, dhat, _, thats = fits[0]
    assert abs(khat[0, 0]/K[0, 0]-1) < .03
    assert abs(khat[1, 1]/K[1, 1]-1) < .03
    np.testing.assert_allclose(fits[1][1], khat, rtol=1e-4, atol=1e-3)
    ratios = np.array([b.ravel()/a.ravel() for a, b in zip(thats, fits[1][4])])
    np.testing.assert_allclose(ratios, 1.05, rtol=1e-4, atol=1e-4)
    rv_hold = np.array([.32, -.28, .08])
    tv_hold = np.array([-.06, -.05, .72])
    uv_true, _ = cv2.projectPoints(board, rv_hold, tv_hold, K, distortion)
    uv_hat, _ = cv2.projectPoints(board, rv_hold, tv_hold, khat, dhat)
    uv_no_dist, _ = cv2.projectPoints(board, rv_hold, tv_hold, khat, np.zeros(5))
    residual = np.linalg.norm(uv_hat-uv_true, axis=2).ravel()
    residual_no_dist = np.linalg.norm(uv_no_dist-uv_true, axis=2).ravel()
    report = {
        'metadata': {'verified_date': '2026-09-22', 'seed': 1313,
                     'python': platform.python_version(), 'numpy': np.__version__,
                     'opencv': cv2.__version__, 'matplotlib': matplotlib.__version__,
                     'device': 'CPU', 'length_unit': 'm', 'image_unit': 'px',
                     'training_steps': 0, 'robot_interactions': 0,
                     'elapsed_ms': (time.perf_counter()-begin)*1000},
        'stereo': {'shape_DHW': list(volume.shape), 'true_disparity_px': disparity,
                   'valid_roi_pixels': int(hard[roi].size), 'exact_match_fraction': accuracy,
                   'cost_curve': curve.tolist(), 'temperature_cases': soft_cases,
                   'two_peak_expected_disparity_px': mean,
                   'two_peak_probability_at_expected_disparity': float(prob[6]),
                   'gradient_max_difference': float(np.max(abs(derivative-numeric)))},
        'calibration': {'views': 20, 'points_per_view': 54, 'corner_sigma_px': .12,
                        'K_true': K.tolist(), 'K_estimated': khat.tolist(),
                        'dist_true': distortion.tolist(), 'dist_estimated': dhat.tolist(),
                        'fit_rms_px': rms, 'wrong_square_size_rms_px': fits[1][0],
                        'wrong_square_size_translation_ratio_mean': float(ratios.mean()),
                        'heldout_true_pose_rms_px': float(np.sqrt(np.mean(residual**2))),
                        'heldout_without_distortion_rms_px': float(np.sqrt(np.mean(residual_no_dist**2))),
                        'holdout_protocol': 'New pose and clean pixels; known synthetic pose fixed, not refit',
                        'radial_terms_fitted': ['k1', 'k2'], 'tangential_terms': 'fixed zero'}
    }
    (out/'stereo-calibration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for case in soft_cases:
        axes[0].plot(np.arange(13), case['probabilities'], marker='.',
                     label=f"temperature={case['temperature']}")
    axes[0].axvline(6, ls='--', color='k', alpha=.5)
    axes[0].set(title='Same costs, different temperatures', xlabel='Disparity (px)', ylabel='Weight')
    axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(13), prob, color='#2563eb')
    axes[1].axvline(mean, color='#dc2626', ls='--', label='Mean = 6, no supported mode')
    axes[1].set(title='A mean can fall between two modes', xlabel='Disparity (px)')
    axes[1].legend(fontsize=8)
    axes[2].plot(residual, label='Estimated K + estimated distortion')
    axes[2].plot(residual_no_dist, label='Same K, distortion ignored')
    axes[2].set(title='Held-out pose; clean synthetic pixels', xlabel='Board corner index', ylabel='Reprojection error (px)')
    axes[2].legend(fontsize=8)
    fig.suptitle('D13 | Original-plan stereo and full calibration experiments')
    fig.savefig(out/'stereo-calibration.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'stereo_accuracy': accuracy, 'calibration': report['calibration']}, indent=2))


if __name__ == '__main__':
    main()
