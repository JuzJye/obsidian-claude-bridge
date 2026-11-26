#!/usr/bin/env python3
"""D69: causal multimodal identification + closed-state synthetic rollouts.
No network, hardware, real RGB/depth image, or learned contact physics.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DT, MASS, DAMP, HISTORY, HORIZON = .02, .5, 1., 6, 10
MODES = {'V': (False, False), 'V+D': (True, False),
         'V+T': (False, True), 'V+D+T': (True, True)}
LAMBDAS = [1e-5, 1e-4, 1e-3]  # m^2; declared before test inspection


def step(z, u):
    """z[...,3]=[x(m),v(m/s),k(N/m)], u[...] in newtons."""
    x, v, k = np.moveaxis(z, -1, 0)
    vn = v + DT / MASS * (u - DAMP * v - k * np.maximum(x, 0))
    return np.stack((x + DT * vn, vn, k), axis=-1)


def rollout(z0, actions):
    """Only estimated state and predeclared actions. No observations/truth."""
    z, out = z0.copy(), []
    for u in actions.T:
        z = step(z, u)
        out.append(z.copy())
    return np.stack(out, axis=1)


def make_data(seed, count, krange):
    rng = np.random.default_rng(seed)
    z0 = np.column_stack((rng.uniform(-.012, .022, count),
                          rng.uniform(-.06, .06, count), rng.uniform(*krange, count)))
    # 15 intervals; five force commands, each held for three intervals.
    actions = np.repeat(rng.uniform(-1, 1, (count, 5)), 3, axis=1)
    truth = np.concatenate((z0[:, None], rollout(z0, actions)), axis=1)
    x, v, k = np.moveaxis(truth, -1, 0)
    obs = np.stack((x + rng.normal(0, .003, x.shape),
                    v + rng.normal(0, .006, v.shape),
                    x + rng.normal(0, .0003, x.shape),
                    k * np.maximum(x, 0) + rng.normal(0, .015, x.shape)), axis=-1)
    return {'truth': truth, 'obs': obs, 'actions': actions}


def estimate(history, depth, tactile, prior, lam):
    assert history.shape[1:] == (HISTORY, 4)
    pos = history[:, :, 0]
    if depth:
        pos = (pos / .003**2 + history[:, :, 2] / .0003**2) / (1/.003**2 + 1/.0003**2)
    p = np.maximum(pos, 0)
    k = np.full(len(pos), prior)
    if tactile:
        k = (np.sum(p * history[:, :, 3], axis=1) + lam * prior) / (np.sum(p*p, axis=1) + lam)
        k = np.clip(k, 5., 200.)  # declared broad physical bounds, not training extrema
    return np.column_stack((pos[:, -1], history[:, -1, 1], k))


def predict(history, actions, mode, params):
    depth, tactile = MODES[mode]
    z0 = estimate(history, depth, tactile, params['prior'], params['lambda'][mode])
    return z0, rollout(z0, actions)


def inputs(data, bias=0.):
    hist = data['obs'][:, :HISTORY].copy()
    hist[:, :, 3] += bias
    return hist, data['actions'][:, HISTORY-1:].copy()


def fit(train, val):
    params = {'prior': float(train['truth'][:, 0, 2].mean()), 'lambda': {}, 'val_scores': {}}
    hist, actions = inputs(val)
    target = val['truth'][:, HISTORY:, :2]
    for mode, (d, t) in MODES.items():
        candidates = LAMBDAS if t else [LAMBDAS[0]]
        scores = []
        for lam in candidates:
            pred = rollout(estimate(hist, d, t, params['prior'], lam), actions)
            scores.append(float(np.mean(((pred[:, -1, :2]-target[:, -1]) / [.01, .1])**2)))
        params['lambda'][mode] = candidates[int(np.argmin(scores))]
        params['val_scores'][mode] = scores
    return params


def metrics(pred, truth):
    e = pred[:, :, :2] - truth[:, :, :2]
    rmse = np.sqrt(np.mean(e*e, axis=0)) * 1000
    prefix = np.sqrt(np.cumsum(np.mean(e*e, axis=0), axis=0) / np.arange(1, HORIZON+1)[:, None]) * 1000
    return {'endpoint_x_mm': rmse[:, 0].tolist(), 'endpoint_v_mm_s': rmse[:, 1].tolist(),
            'prefix_x_mm': prefix[:, 0].tolist(), 'prefix_v_mm_s': prefix[:, 1].tolist(),
            'contact_correct': ((pred[:, :, 0] > 0) == (truth[:, :, 0] > 0)).sum(axis=0).tolist(),
            'episodes': len(pred)}


def evaluate(data, params, bias):
    hist, actions = inputs(data, bias)
    truth = data['truth'][:, HISTORY:]
    results, saved = {}, {}
    for mode in MODES:
        z0, pred = predict(hist, actions, mode, params)
        results[mode] = metrics(pred, truth)
        results[mode]['initial_rmse_x_v_k'] = np.sqrt(np.mean((z0-data['truth'][:, HISTORY-1])**2, axis=0)).tolist()
        saved[mode] = pred
    # A diagnostic with privileged state resets; NOT a deployable rollout.
    prev = data['truth'][:, HISTORY-1:-1].copy()
    prev[:, :, 2] = estimate(hist, True, True, params['prior'], params['lambda']['V+D+T'])[:, None, 2]
    oracle = step(prev, actions)
    results['oracle_reset'] = metrics(oracle, truth)
    saved['oracle_reset'] = oracle
    return results, saved


def latency(history, actions, params):
    for _ in range(10):
        predict(history, actions, 'V+D+T', params)
    times = []
    for _ in range(100):
        t = time.perf_counter()
        predict(history, actions, 'V+D+T', params)
        times.append((time.perf_counter()-t)*1000)
    return {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
            'batch': len(history), 'warmups': 10, 'repeats': 100,
            'scope': 'initial observer + 10-step rollout; excludes IO and metrics'}


def figures(out, report, saved, test, bias):
    h = np.arange(1, HORIZON+1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, split in enumerate(('test', 'ood')):
        for mode in MODES:
            r = report[split][mode]
            axes[row, 0].plot(h, r['endpoint_x_mm'], '-o', ms=3, label=mode)
            axes[row, 1].plot(h, r['endpoint_v_mm_s'], '-o', ms=3, label=mode)
        for col, unit in enumerate(('position RMSE (mm)', 'velocity RMSE (mm/s)')):
            axes[row, col].set(xlabel='Future step (dt=0.02 s)', ylabel=unit,
                                title=f'{split.upper()}: endpoint error, n=128')
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=8)
    fig.suptitle(f'Closed-state free rollout | tactile history bias = {bias:g} N')
    fig.savefig(out/'rollout_errors.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for mode in MODES:
        axes[0].plot(h, report['test'][mode]['prefix_x_mm'], label=mode)
    axes[0].set(xlabel='Prefix length H', ylabel='Prefix position RMSE (mm)', title='All steps 1..H, same 128 episodes')
    axes[0].legend(fontsize=8)
    for mode, style in (('V+D+T', '-o'), ('oracle_reset', '--s')):
        axes[1].plot(h, report['test'][mode]['endpoint_v_mm_s'], style, label=mode)
    axes[1].set(xlabel='Future step', ylabel='Velocity RMSE (mm/s)', title='Truth resets change the question')
    axes[1].legend(fontsize=8)
    # Episode 0 is predeclared, never chosen by nicest/worst outcome.
    axes[2].plot(h*DT, test['truth'][0, HISTORY:, 0]*1000, 'k-o', label='truth')
    for mode in MODES:
        axes[2].plot(h*DT, saved[mode][0, :, 0]*1000, label=mode)
    axes[2].axhline(0, c='gray', lw=1)
    axes[2].set(xlabel='Future time (s)', ylabel='Signed compression x (mm)', title='Fixed test episode 0; contact x>0')
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'protocol_and_example.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--touch-bias', type=float, default=0., help='N; only test/OOD history')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day69/default')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = {name: make_data(seed, n, kr) for name, seed, n, kr in
            [('train', 690, 256, (20.,80.)), ('val', 691, 128, (20.,80.)),
             ('test', 692, 128, (20.,80.)), ('ood', 693, 128, (100.,150.))]}
    params = fit(data['train'], data['val'])
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'touch_bias_N': args.touch_bias,
              'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
              'platform': platform.platform(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'parameters': params, 'config': {'dt_s': DT, 'mass_kg': MASS, 'damping_N_s_m': DAMP,
              'history_frames': HISTORY, 'future_steps': HORIZON, 'seeds': [690,691,692,693],
              'counts': [256,128,128,128], 'units': ['m','m/s','N/m','N'], 'physical_robot_trials': 0,
              'history_times_s': (np.arange(-5,1)*DT).tolist(), 'initial_x_m': [-.012,.022],
              'initial_v_m_s': [-.06,.06], 'train_val_test_k_N_m': [20,80], 'ood_k_N_m': [100,150],
              'noise_sd_vx_vv_dx_tf': [.003,.006,.0003,.015], 'action_range_N': [-1,1],
              'action_hold_steps': 3, 'k_clip_N_m': [5,200], 'lambda_candidates_m2': LAMBDAS,
              'train_intervals': 3840, 'val_intervals': 1920, 'test_intervals': 1920, 'ood_intervals': 1920}}
    arrays = {}
    for split in ('test', 'ood'):
        report[split], predictions = evaluate(data[split], params, args.touch_bias)
        for name, value in predictions.items(): arrays[split+'_'+name] = value
        for name, value in data[split].items(): arrays[split+'_'+name] = value
        if split == 'test': test_saved = predictions
    hist, actions = inputs(data['test'], args.touch_bias)
    arrays['test_input_history'], arrays['test_future_actions'] = hist, actions
    arrays['ood_input_history'], arrays['ood_future_actions'] = inputs(data['ood'], args.touch_bias)
    altered = {k: v.copy() for k, v in data['test'].items()}
    altered['truth'][:] = 999.  # including privileged k
    altered['obs'][:, HISTORY:] = -999.
    ha, aa = inputs(altered, args.touch_bias)
    for mode in MODES:
        assert np.array_equal(predict(hist, actions, mode, params)[1], predict(ha, aa, mode, params)[1])
    assert np.max(np.abs(rollout(data['test']['truth'][:, HISTORY-1], actions)-data['test']['truth'][:, HISTORY:])) < 1e-12
    report['checks'] = {'future_truth_and_observation_invariance': True, 'oracle_initial_state_reproduces_fixture': True}
    report['latency'] = [latency(hist[:1], actions[:1], params), latency(hist, actions, params)]
    report['model_sha256'] = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    np.savez_compressed(args.out/'predictions.npz', **arrays)
    (args.out/'model.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    (args.out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    figures(args.out, report, test_saved, data['test'], args.touch_bias)
    for mode in MODES:
        r = report['test'][mode]
        print(mode, 'x_mm@1/5/10=', np.array(r['endpoint_x_mm'])[[0,4,9]].round(6),
              'v_mm_s@1/5/10=', np.array(r['endpoint_v_mm_s'])[[0,4,9]].round(6))
    print('lambda:', params['lambda'], 'causality checks passed; physical robot trials: 0')


if __name__ == '__main__':
    main()
