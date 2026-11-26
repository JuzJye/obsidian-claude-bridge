"""D36: synthetic protocol manifest and denominator audit; no dataset download."""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validate(record):
    errors = []
    if record['length_unit'] != 'm':
        errors.append('length_unit')
    if record['pose_direction'] != 'object_to_camera':
        errors.append('pose_direction')
    r = np.asarray(record['R'], float)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        errors.append('SO3')
    if not (0 <= record['visibility'] <= 1):
        errors.append('visibility')
    if len(record['t']) != 3 or record['t'][2] <= 0:
        errors.append('positive_depth')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-visibility', type=float, default=0.0)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2] / '_assets/3d_learning/day36')
    args = parser.parse_args()
    if not 0 <= args.min_visibility <= 1:
        raise ValueError('min visibility must be in [0,1]')
    started = time.perf_counter()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(36)
    rows = []
    for i in range(120):
        v = (i % 10 + 0.5) / 10
        # Deliberately create harder low-visibility targets, not a learned model.
        emitted = (i % 10 >= 2) or (i % 3 == 0)
        error_m = (0.004 + 0.040 * (1 - v) ** 2 + rng.uniform(0, .004)) if emitted else None
        rows.append(dict(scene_id=i // 10, frame_id=i % 10, obj_id=i % 3,
                         target_id=f'{i//10:03d}/{i%10:03d}/{i%3}',
                         length_unit='m', pose_direction='object_to_camera',
                         R=np.eye(3).tolist(), t=[.02, -.01, .8],
                         visibility=v, diameter_m=.20, emitted=emitted,
                         add_m=error_m))
    assert not any(validate(r) for r in rows)
    assert len({r['target_id'] for r in rows}) == len(rows)
    threshold_m = .02
    selected = [r for r in rows if r['visibility'] >= args.min_visibility]
    if not selected:
        raise ValueError('no targets after visibility selection')
    n = len(selected)
    emitted = sum(r['emitted'] for r in selected)
    passed = sum(r['emitted'] and r['add_m'] < threshold_m for r in selected)
    corruptions = []
    import copy
    for label in ['length_unit', 'pose_direction', 'SO3', 'visibility', 'positive_depth']:
        r = copy.deepcopy(rows[0])
        if label == 'length_unit': r['length_unit'] = 'mm'
        if label == 'pose_direction': r['pose_direction'] = 'camera_to_object'
        if label == 'SO3': r['R'][0][0] = -1
        if label == 'visibility': r['visibility'] = 1.2
        if label == 'positive_depth': r['t'][2] = -.8
        corruptions.append({'fault': label, 'detected': validate(r)})
        assert label in validate(r)
    serialized = json.dumps(rows, sort_keys=True).encode()
    result = dict(day=36, verified_on='2026-10-24', seed=36, device='CPU',
                  python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                  data='120 synthetic single-object targets; no real images', training_steps=0,
                  target_count=n, emitted_count=emitted, success_count=passed,
                  add_threshold_m=threshold_m, coverage=emitted/n,
                  conditional_success=passed/emitted if emitted else None,
                  all_target_success=passed/n, min_visibility=args.min_visibility,
                  manifest_sha256=hashlib.sha256(serialized).hexdigest(),
                  schema_negative_cases=corruptions, hardware_interactions=0,
                  elapsed_s=time.perf_counter()-started)
    (out/'manifest.json').write_text(json.dumps(rows, indent=2))
    (out/'results.json').write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    bins = [(0, .3), (.3, .7), (.7, 1.01)]
    counts = []
    for lo, hi in bins:
        group = [r for r in selected if lo <= r['visibility'] < hi]
        counts.append([len(group), sum(r['emitted'] for r in group),
                       sum(r['emitted'] and r['add_m'] < threshold_m for r in group)])
    a = np.asarray(counts)
    x = np.arange(3)
    for j, name in enumerate(['targets','outputs','passed']):
        ax[0].bar(x+(j-1)*.24, a[:,j], .24, label=name)
    ax[0].set_xticks(x, ['low','medium','high']); ax[0].set_xlabel('Visibility bin')
    ax[0].set_ylabel('Count'); ax[0].legend(); ax[0].set_title('Every target remains accountable')
    rates = [result['coverage'], result['conditional_success'] or 0, result['all_target_success']]
    ax[1].bar(['coverage','given output','all targets'], rates, color=['#5778a4','#e49444','#4c998b'])
    for i,v in enumerate(rates): ax[1].text(i,v+.02,f'{v:.1%}',ha='center')
    ax[1].set_ylim(0,1.15); ax[1].set_ylabel('Fraction'); ax[1].set_title('Denominator changes the claim')
    fig.savefig(out/'protocol.png', dpi=160); plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
