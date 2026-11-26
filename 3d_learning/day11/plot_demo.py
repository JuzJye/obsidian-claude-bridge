"""Plot saved results; no recomputation or external data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day11'
a=np.load(root/'arrays.npz'); r=json.loads((root/'results.json').read_text())
fig,ax=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
for axis,key,title in zip(ax.ravel(),['left','right_offset0','sad1_offset0','sad7_offset0','sgbm_offset0','sad7_offset30'],['Left image','Right image','Pixel cost / WTA','7x7 mean cost / WTA','SGBM','7x7 / right +30']):
    if 'left'==key or 'right' in key:
        axis.imshow(a[key],cmap='gray',vmin=0,vmax=255)
    else:
        im=axis.imshow(a[key],cmap='viridis',vmin=0,vmax=23)
    axis.set_title(title);axis.set_xlabel('u (px)');axis.set_ylabel('v (px)')
fig.colorbar(im,ax=ax[:,1:].ravel().tolist(),label='Disparity (px)',shrink=.65)
fig.savefig(root/'stereo-maps.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
for key,label in [('cost_textured','Textured'),('cost_flat','Textureless')]:ax[0].plot(np.arange(24),a[key],label=label)
ax[0].axvline(8,color='k',ls='--',label='Truth = 8');ax[0].legend();ax[0].set(xlabel='Disparity candidate (px)',ylabel='Mean absolute difference',title='Cost ambiguity')
ax[1].plot(a['depth'],a['sigma_z']*1000,'o-');ax[1].set(xlabel='Depth Z (m)',ylabel='Linearized depth sigma (mm)',title='Fixed disparity sigma = 0.25 px');ax[1].grid(alpha=.25)
fig.savefig(root/'cost-depth.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
for i,label in enumerate(['Train','Validation']):ax[0].semilogy(np.arange(1,61),a['loss_history'][:,i],label=label)
ax[0].legend();ax[0].set(xlabel='Epoch',ylabel='MSE (m^2)',title='Affine calibration learned by batches')
keys=['random_frames','grouped_scenes']; vals=[r['leakage'][k]['rmse'] for k in keys]
ax[1].bar(['Random frames','Held-out scenes'],vals,color=['#419d78','#d36135']);ax[1].set(ylabel='Test RMSE (synthetic label units)',title='1-NN scene memorization')
for i,v in enumerate(vals):ax[1].text(i,v+.008,f'{v:.3f}',ha='center')
ax[1].set_ylim(0,.36);fig.savefig(root/'learning-split.png',dpi=170);plt.close(fig)
print('Saved stereo-maps.png, cost-depth.png, learning-split.png')
