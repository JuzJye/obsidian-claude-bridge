"""Day12 figures generated entirely from saved synthetic calibration evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day12'
a=np.load(root/'arrays.npz');r=json.loads((root/'results.json').read_text())['results']
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for i,p in enumerate(a['clean']):
    ax[0].scatter(p[:,0],p[:,1],s=3,alpha=.4,color=plt.cm.viridis(i/24))
ax[0].set(xlim=(0,640),ylim=(480,0),xlabel='u (px)',ylabel='v (px)',title='24 projected board views; 18 train + 6 holdout');ax[0].set_aspect('equal')
x=a['observed'][:18].reshape(-1,2);err=a['omitted_residuals'].reshape(-1,2)
ax[1].quiver(x[::5,0],x[::5,1],err[::5,0],err[::5,1],angles='xy',scale_units='xy',scale=.06,width=.004)
ax[1].set(xlim=(0,640),ylim=(480,0),xlabel='u (px)',ylabel='v (px)',title='Residuals when distortion is forced to zero (x16.7)');ax[1].set_aspect('equal')
fig.savefig(root/'calibration-residuals.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
names=['Training fit','Heldout, fixed pose','Heldout, fitted pose'];x=np.arange(3)
for offset,sigma in [(-.18,.2),(.18,1.)]:
    q=r[f'noise_{sigma}']; vals=[q['training_rms_px'],q['heldout_fixed_truth_pose_rms_px'],q['heldout_fitted_pose_rms_px']]
    ax[0].bar(x+offset,vals,.34,label=f'Corner sigma {sigma} px')
ax[0].set_xticks(x,names,rotation=12);ax[0].set(ylabel='2D point RMS (px)',title='Different evidence checks different unknowns');ax[0].legend()
w=r['workspace'];x=np.arange(4)
ax[1].bar(x-.18,np.array(w['focal_plus5_error_m'])*1000,.34,label='f +5%')
ax[1].bar(x+.18,np.array(w['extrinsic_y_axis1deg_error_m'])*1000,.34,label='Extrinsic rotation 1 deg about y')
ax[1].set_xticks(x,['Center,0.5m','Off-axis,0.5m','Off-axis,1m','Farther ray,1m'],rotation=18);ax[1].set(ylabel='Position error (mm)',title='Workspace error with independent depth');ax[1].legend(fontsize=8)
fig.savefig(root/'parameter-workspace.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True)
for history,label in zip(a['feedback_error'],['Unbiased feedback','Feedback bias +10 mm']):ax.plot(np.arange(16),history*1000,'o-',label=label)
ax.axhline(0,color='k',lw=.8);ax.axhline(-10,color='gray',ls='--');ax.set(xlabel='Feedback update',ylabel='True target error (mm)',title='Feedback removes initial error, not an unknown sensor bias');ax.legend();ax.grid(alpha=.2)
fig.savefig(root/'feedback-bias.png',dpi=170);plt.close(fig)
print('Saved calibration-residuals.png, parameter-workspace.png, feedback-bias.png')
