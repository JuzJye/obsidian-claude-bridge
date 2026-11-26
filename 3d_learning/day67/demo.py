"""CPU protocol experiment: synthetic SE(3) predictions, not a trained pose model."""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rotvec(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    x, y, z = w / a
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def bce(logit, y):
    return float(np.mean(np.logaddexp(0, logit) - y * logit))


def classification(p, y, threshold):
    q = p >= threshold
    tp, fp = int(np.sum(q & (y == 1))), int(np.sum(q & (y == 0)))
    fn, tn = int(np.sum(~q & (y == 1))), int(np.sum(~q & (y == 0)))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, count=len(y),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                accuracy=(tp + tn) / len(y), cost_5fn_fp=5 * fn + fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=67)
    ap.add_argument('--drop-hard', type=float, default=0)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / '_assets/3d_learning/day67/default')
    args = ap.parse_args()
    if not 0 <= args.drop_hard < 1:
        ap.error('--drop-hard must be in [0, 1)')
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    model = np.array([[x, y, z] for x in [-.04, .04] for y in [-.025, .025] for z in [-.02, .02]])
    sym = [np.eye(3), rotvec(np.array([0., 0., np.pi]))]
    n = 240
    gt_r = np.array([rotvec(rng.normal(size=3)) for _ in range(n)])
    gt_t = np.column_stack([rng.uniform(-.1, .1, n), rng.uniform(-.1, .1, n), rng.uniform(.4, .8, n)])
    pred_r = np.array([rotvec(rng.normal(scale=np.deg2rad(3), size=3)) @ r @ sym[int(rng.random() < .25)] for r in gt_r])
    pred_t = gt_t + rng.normal(scale=.004, size=(n, 3))
    pred_t[:48] += rng.normal(scale=.025, size=(48, 3))
    add, adds, symmetry_add, mssd, rotation = [], [], [], [], []
    for rg, tg, rp, tp in zip(gt_r, gt_t, pred_r, pred_t):
        truth, prediction = model @ rg.T + tg, model @ rp.T + tp
        add.append(np.linalg.norm(prediction - truth, axis=-1).mean())
        adds.append(np.linalg.norm(prediction[:, None] - truth[None, :], axis=-1).min(1).mean())
        distances = [np.linalg.norm(prediction - (model @ (rg @ s).T + tg), axis=-1) for s in sym]
        symmetry_add.append(min(d.mean() for d in distances))
        mssd.append(min(d.max() for d in distances))
        rotation.append(np.rad2deg(np.arccos(np.clip((np.trace(rp.T @ rg) - 1) / 2, -1, 1))))
    add, adds, symmetry_add, mssd, rotation = map(np.asarray, (add, adds, symmetry_add, mssd, rotation))
    # Independent validation examples fit temperature and operating threshold.
    eta_val, eta_test = rng.normal(-.7, 1.8, 400), rng.normal(-.7, 1.8, n)
    yv, yt = rng.binomial(1, sigmoid(eta_val)), rng.binomial(1, sigmoid(eta_test))
    lv, lt = 3 * eta_val, 3 * eta_test
    temperatures = np.linspace(.5, 6, 111)
    temperature = float(temperatures[np.argmin([bce(lv / t, yv) for t in temperatures])])
    candidates = np.linspace(.05, .95, 91)
    threshold = float(candidates[np.argmin([classification(sigmoid(lv / temperature), yv, t)['cost_5fn_fp'] for t in candidates])])
    p = sigmoid(lt / temperature)
    available = np.ones(n, bool)
    # Deliberately biased omission; benchmark targets remain in the denominator.
    available[np.argsort(mssd)[::-1][:int(n * args.drop_hard)]] = False
    correct = mssd < .01  # fixed task tolerance, not tuned on test results
    contact_correct = (p >= threshold) == yt
    retained = int(available.sum())
    exact_half_turn = np.linalg.norm(model @ sym[1].T - model, axis=-1).mean()
    equivalent_error = min(np.linalg.norm(model @ sym[1].T - model @ s.T, axis=-1).mean() for s in sym)
    translation_check = np.linalg.norm((model + [.004, 0, 0]) - model, axis=-1).mean()
    assert exact_half_turn > .09 and np.allclose(np.linalg.det(pred_r), 1)
    assert equivalent_error < 1e-12 and np.isclose(translation_check, .004)
    assert np.max(np.abs(pred_r.transpose(0, 2, 1) @ pred_r - np.eye(3))) < 1e-12
    result = dict(seed=args.seed, created_at=datetime.now(timezone.utc).isoformat(),
                  data='synthetic SE(3) prediction fixture and Bernoulli contact labels; no trained pose network',
                  units='model and translation: metre; rotation: SO(3); displayed distance: mm',
                  targets=n, returned=retained, missing=n-retained, coverage=retained/n,
                  pose=dict(tolerance_mm=10, available_correct=int((correct & available).sum()),
                            recall_full=float((correct & available).sum()/n),
                            conditional_success=float(correct[available].mean()),
                            mean_ADD_mm=float(add[available].mean()*1000),
                            mean_ADD_S_mm=float(adds[available].mean()*1000),
                            mean_symmetry_ADD_mm=float(symmetry_add[available].mean()*1000),
                            mean_MSSD_mm=float(mssd[available].mean()*1000)),
                  calibration=dict(validation_count=400, temperature=temperature, threshold=threshold,
                                   raw_test_nll=bce(lt, yt), calibrated_test_nll=bce(lt/temperature, yt),
                                   raw_brier=float(np.mean((sigmoid(lt)-yt)**2)), calibrated_brier=float(np.mean((p-yt)**2))),
                  contact_available=classification(p[available], yt[available], threshold),
                  contact_at_half=classification(p[available], yt[available], .5),
                  joint_correct=int((correct & contact_correct & available).sum()),
                  joint_rate=float((correct & contact_correct & available).sum()/n),
                  symmetry_unit_check=dict(raw_ADD_mm=float(exact_half_turn*1000), symmetry_ADD_mm=float(equivalent_error*1000), pure_translation_mm=float(translation_check*1000), raw_angle_deg=180),
                  elapsed_seconds=time.perf_counter()-start, robot_trials=0, project_acceptance='pending',
                  environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__, device='cpu'))
    (args.out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez(args.out/'predictions.npz', gt_r=gt_r, gt_t=gt_t, pred_r=pred_r, pred_t=pred_t,
             add=add, adds=adds, symmetry_add=symmetry_add, mssd=mssd, available=available, y=yt, probability=p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    points = axes[0].scatter(add*1000, symmetry_add*1000, c=rotation, s=15, cmap='viridis')
    fig.colorbar(points, ax=axes[0], label='Raw rotation error (deg)')
    axes[0].plot([0, 150], [0, 150], 'k--', alpha=.5)
    axes[0].set(xlabel='ADD (mm)', ylabel='Symmetry-aware mean distance (mm)', title='Declared half-turn equivalence')
    axes[1].bar(['Coverage', 'Full recall', 'Returned-only'], [retained/n, result['pose']['recall_full'], result['pose']['conditional_success']], color=['#6688aa', '#228899', '#dd7755'])
    axes[1].set(ylim=(0, 1), ylabel='Fraction', title=f'{retained}/{n} predictions returned')
    fig.tight_layout(); fig.savefig(args.out/'pose_metrics.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, prob in [('raw', sigmoid(lt)), ('temperature', p)]:
        xs, ys = [], []
        for lo in np.linspace(0, .9, 10):
            mask = (prob >= lo) & (prob < lo+.1)
            if mask.any(): xs.append(prob[mask].mean()); ys.append(yt[mask].mean())
        axes[0].plot(xs, ys, 'o-', label=name)
    axes[0].plot([0, 1], [0, 1], 'k--'); axes[0].set(xlabel='Mean predicted probability', ylabel='Observed contact fraction', title='Synthetic test reliability'); axes[0].legend()
    costs = [classification(sigmoid(lv/temperature), yv, t)['cost_5fn_fp']/len(yv) for t in candidates]
    axes[1].plot(candidates, costs); axes[1].axvline(threshold, color='red', ls='--')
    axes[1].set(xlabel='Contact decision threshold', ylabel='Validation (5 FN + FP) / N', title='Threshold selected before test')
    fig.tight_layout(); fig.savefig(args.out/'contact_metrics.png', dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
