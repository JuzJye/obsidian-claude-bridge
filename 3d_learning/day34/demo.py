"""Small learned vision-language grounding, geometry contracts, and LoRA.
All data are synthetic. This script is not an openpi inference demo.
"""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/day34-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/day34-cache')
import numpy as np
import torch
from torch import nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / '_assets/3d_learning/day34'
COLORS = np.array([[.9, .10, .08], [.08, .22, .9], [.10, .85, .12]])


def features(crop, mask):
    rgb = crop[mask].mean(0)
    y, x = np.where(mask)
    h, w = y.max()-y.min()+1, x.max()-x.min()+1
    return np.r_[rgb, mask.sum()/(h*w), w/h, mask.sum()/1024.].astype('float32')


def shape_patch(color, shape, radius=13):
    mask = np.zeros((36, 36), np.uint8)
    if shape == 0:
        cv2.circle(mask, (18, 18), radius, 1, -1)
    else:
        cv2.rectangle(mask, (18-radius, 18-radius), (18+radius, 18+radius), 1, -1)
    return mask[..., None] * color, mask.astype(bool)


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        c, shape = rng.integers(3), rng.integers(2)
        color = np.clip(COLORS[c] + rng.normal(0, .035, 3), 0, 1)
        crop, mask = shape_patch(color, shape, int(rng.integers(9, 16)))
        xs.append(features(crop, mask)); ys.append(int(c*2+shape))
    return torch.tensor(np.array(xs)), torch.tensor(ys)


def language_features():
    return torch.tensor(np.array([np.r_[np.eye(3)[c], np.eye(2)[s]]
                                  for c in range(3) for s in range(2)]), dtype=torch.float32)


def grounding():
    train_x, train_y = dataset(768, 340)
    test_x, test_y = dataset(384, 341)
    image_encoder = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 12))
    text_encoder = nn.Linear(5, 12, bias=False)
    opt = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=.01)
    text = language_features()
    def logits(x):
        im = nn.functional.normalize(image_encoder(x), dim=-1)
        tx = nn.functional.normalize(text_encoder(text), dim=-1)
        return im @ tx.T / .1
    for _ in range(600):
        ids = torch.randint(len(train_x), (64,))
        loss = nn.functional.cross_entropy(logits(train_x[ids]), train_y[ids])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        accuracy = float((logits(test_x).argmax(1) == test_y).float().mean())
    scene = np.zeros((120, 160, 3), dtype=float)
    depth = np.zeros((120, 160), dtype=float)
    specifications = [(0, 0, 50, 70, .45), (1, 1, 110, 40, .52), (2, 0, 110, 90, .48)]
    for color, shape, u, v, z in specifications:
        crop, mask = shape_patch(COLORS[color], shape)
        scene[v-18:v+18, u-18:u+18] = crop
        depth[v-18:v+18, u-18:u+18][mask] = z
    binary = (scene.max(-1) > .3).astype('uint8')
    count, labels = cv2.connectedComponents(binary)
    candidates = []
    goal = text[0]  # closed-vocabulary command: red circle
    for instance in range(1, count):
        mask = labels == instance
        y, x = np.where(mask)
        feat = features(scene, mask)
        with torch.no_grad():
            im = nn.functional.normalize(image_encoder(torch.from_numpy(feat)[None]), dim=-1)
            tx = nn.functional.normalize(text_encoder(goal[None]), dim=-1)
            score = float((im * tx).sum())
        candidates.append({'id': instance, 'score': score, 'u': float(x.mean()), 'v': float(y.mean()),
                           'pixel_width': int(x.max()-x.min()+1), 'z': float(np.median(depth[mask]))})
    candidates.sort(key=lambda item: item['score'], reverse=True)
    best = candidates[0]
    semantic_margin = best['score'] - candidates[1]['score']
    point_cam = np.array([(best['u'] - 80)*best['z']/120,
                          (best['v'] - 60)*best['z']/120, best['z']])
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    point_base = R @ point_cam + np.array([.30, .10, .20])
    pregrasp = point_base + np.array([0., 0., .06])
    width = best['pixel_width'] * best['z'] / 120 + .01
    def contract(z, age, margin):
        return bool(.1 <= z <= 2. and age <= .10 and margin >= .10)
    gate_cases = {'nominal': contract(best['z'], .03, semantic_margin),
                  'depth_in_mm': contract(best['z']*1000, .03, semantic_margin),
                  'stale_observation': contract(best['z'], .25, semantic_margin),
                  'ambiguous_goal': contract(best['z'], .03, .01)}
    report = {'test_feature_accuracy': accuracy, 'candidates': candidates,
              'semantic_margin': semantic_margin, 'point_camera_m': point_cam.tolist(),
              'point_base_m': point_base.tolist(), 'pregrasp_base_m': pregrasp.tolist(),
              'contract_cases': gate_cases,
              'proposal': {'frame': 'base', 'mode': 'absolute_pose', 'position_m': pregrasp.tolist(),
                           'quaternion_xyzw': [0., 0., 0., 1.], 'gripper_width_m': width,
                           'observation_age_s': .03, 'execution_authorized': False}}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    ax[0].imshow(scene); ax[0].scatter([best['u']], [best['v']], marker='+', c='white', s=150)
    ax[0].set_title('Learned grounding: red circle')
    ax[1].imshow(depth, vmin=0, vmax=.6, cmap='viridis'); ax[1].set_title('Aligned synthetic depth (m)')
    ax[2].bar([str(c['id']) for c in candidates], [c['score'] for c in candidates], color='#21899b')
    ax[2].set(title='Language-image cosine similarity', xlabel='Detected component ID', ylabel='Score (not calibrated probability)')
    fig.savefig(OUT / 'grounded_pipeline.png', dpi=160); plt.close(fig)
    return report


def lora_experiment():
    gen = torch.Generator().manual_seed(342)
    x = torch.randn(768, 16, generator=gen)
    base = torch.randn(8, 16, generator=gen) * .15
    delta = .12 * torch.randn(8, 2, generator=gen) @ torch.randn(2, 16, generator=gen)
    y = x @ (base + delta).T + .005 * torch.randn(768, 8, generator=gen)
    results, histories = {}, {}
    results['frozen'] = {'trainable': 0, 'test_mse': float(((x[512:] @ base.T - y[512:])**2).mean())}
    for rank in [1, 2, 4]:
        torch.manual_seed(3400 + rank)
        frozen = nn.Parameter(base.clone(), requires_grad=False)
        A = nn.Parameter(.05 * torch.randn(rank, 16))
        B = nn.Parameter(torch.zeros(8, rank))
        optimizer = torch.optim.Adam([A, B], lr=.03)
        history, first_gradient = [], None
        for step in range(700):
            prediction = x[:512] @ (frozen + B @ A).T  # alpha/r=1
            loss = (prediction - y[:512]).square().mean()
            optimizer.zero_grad(); loss.backward()
            if step == 0:
                first_gradient = {'A_norm': float(A.grad.norm()), 'B_norm': float(B.grad.norm()),
                                  'frozen_grad_is_none': frozen.grad is None}
            optimizer.step()
            if step % 20 == 0:
                history.append([step, float(loss.detach())])
        with torch.no_grad():
            merged = frozen + B @ A
            mse = float(((x[512:] @ merged.T - y[512:])**2).mean())
            merge_error = float((x[512:] @ merged.T - (x[512:] @ frozen.T + (x[512:] @ A.T) @ B.T)).abs().max())
        results[f'rank_{rank}'] = {'trainable': A.numel()+B.numel(), 'test_mse': mse,
                                  'base_max_change': float((frozen-base).abs().max()),
                                  'merge_max_error': merge_error, 'first_gradient': first_gradient}
        histories[str(rank)] = np.array(history)
    full = nn.Parameter(base.clone())
    optimizer = torch.optim.Adam([full], lr=.03)
    for _ in range(700):
        loss = (x[:512] @ full.T-y[:512]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    results['full'] = {'trainable': full.numel(), 'test_mse': float(((x[512:] @ full.T-y[512:])**2).mean().detach())}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for rank, history in histories.items():
        ax[0].semilogy(history[:, 0], history[:, 1], label='rank '+rank)
    ax[0].set(xlabel='Optimization step', ylabel='Training MSE', title='Same data and frozen base; varying rank')
    ax[0].legend()
    names = ['frozen', 'rank_1', 'rank_2', 'rank_4', 'full']
    ax[1].bar(names, [results[k]['test_mse'] for k in names], color='#21899b')
    ax[1].set(yscale='log', ylabel='Held-out MSE', title='Synthetic target update has rank 2')
    fig.savefig(OUT / 'lora_learning.png', dpi=160); plt.close(fig)
    np.savez_compressed(OUT / 'lora_data.npz', inputs=x.numpy(), targets=y.numpy(), base=base.numpy(), true_update=delta.numpy())
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(34)
    report = {'verified_date': '2026-10-09', 'torch': torch.__version__, 'numpy': np.__version__,
              'grounding': grounding(), 'lora': lora_experiment()}
    (OUT / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
