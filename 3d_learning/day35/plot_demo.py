"""Plot D35 results without rerunning experiments."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[2] / '_assets/3d_learning/day35'
a = np.load(out / 'arrays.npz')
r = json.loads((out / 'results.json').read_text())
fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
ax[0].plot(a['pnp_history'], marker='o', ms=3)
ax[0].set(xlabel='Attempt / final state', ylabel='Coordinate RMSE / px', title='Handwritten PnP: training objective')
ax[1].hist(a['pnp_test_error'], bins=14, color='#207c91')
ax[1].set(xlabel='Euclidean projection error / px', ylabel='Held-out point count', title='120 unseen 3D points; same camera')
ax[2].plot(a['ba_history'][:, 0], a['ba_history'][:, 2], color='#d27844')
ax[2].set(xlabel='Update attempt', ylabel='Held-out coordinate RMSE / px', title='BA: two exact metric anchors')
fig.savefig(out / 'module-audit.png', dpi=160)
plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
curves = a['continual_losses']
ax[0].plot(curves[:, 0], label='Task A: y=x')
ax[0].plot(curves[:, 1], label='Task B: y=-x')
ax[0].axvline(59.5, color='grey', linestyle='--', label='Switch to task B')
ax[0].set(xlabel='Gradient step', ylabel='Population MSE', title='Sequential data; a shared scalar weight')
ax[0].legend()
ax[1].bar([0, 1, 2], [curves[-1, 0], 1., 0.], color=['#d27844', '#207c91', '#339277'])
ax[1].set_xticks([0, 1, 2], ['After B: A loss', 'Balanced replay\n(each task)', 'Known task ID\n(two heads)'])
ax[1].set(ylabel='Population MSE', title='Data access and task information differ')
ax[1].text(.5, .95, 'No input task ID: opposite labels conflict.\nExtra task ID changes available information.',
           ha='center', va='top', transform=ax[1].transAxes, fontsize=9)
ax[1].set_ylim(0, 5.2)
fig.savefig(out / 'learning-boundaries.png', dpi=160)
plt.close(fig)
print('Wrote module-audit.png and learning-boundaries.png')
