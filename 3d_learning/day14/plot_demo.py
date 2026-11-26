"""Day14 evidence plots from saved arrays only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day14'
a=np.load(root/'arrays.npz');r=json.loads((root/'results.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
errors=(a['depth_samples']-.8)*1000
ax[0].hist(errors,bins=55,density=True,color='#488aa8',alpha=.8);ax[0].set(xlabel='Depth error (mm)',ylabel='Density',title='Stereo uncertainty, disparity sigma = 0.25 px')
g=r['geometry'];x=np.arange(2)
ax[1].bar(x-.17,g['regular_singular_values'],.34,label='Bent arm')
ax[1].bar(x+.17,g['straight_arm_singular_values'],.34,label='Straight arm')
ax[1].set_xticks(x,['Singular value 1','Singular value 2']);ax[1].set(ylabel='Jacobian singular value (m/rad)',title='One velocity direction vanishes at singularity');ax[1].legend()
fig.savefig(root/'geometry-audit.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
names=['zero_shot','bias_only','full_finetune','scratch']
for name in names:ax[0].semilogy(a[f'loss_{name}'],label=name.replace('_',' '))
ax[0].legend(fontsize=8);ax[0].set(xlabel='Target update',ylabel='Train MSE (synthetic units^2)',title='Same target data; change initialized / updated parameters')
vals=[r['transfer'][name]['target_test_rmse'] for name in names]
ax[1].bar(np.arange(4),vals,color=['#aaa','#c77','#488aa8','#419d78']);ax[1].set_xticks(np.arange(4),[s.replace('_','\n') for s in names]);ax[1].set(ylabel='Independent target RMSE',title='Small convex model: scratch can catch up')
for i,v in enumerate(vals):ax[1].text(i,v+.008,f'{v:.3f}',ha='center')
ax[1].set_ylim(0,.46);fig.savefig(root/'parameter-transfer.png',dpi=170);plt.close(fig)
print('Saved geometry-audit.png, parameter-transfer.png')
