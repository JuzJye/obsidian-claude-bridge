"""D18 original-plan supplement: center decoding and nonadjacent self collision.

Synthetic heads and planar geometry, not pretrained CenterNet or a robot check.
"""
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def point_segment_distance(p, a, b):
    ab = b-a
    return float(np.linalg.norm(p-(a+np.clip((p-a)@ab/(ab@ab), 0, 1)*ab)))


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def segment_distance(a, b, c, d):
    ab, cd = b-a, d-c
    den = cross(ab, cd)
    if abs(den) > 1e-12:
        t, u = cross(c-a, cd)/den, cross(c-a, ab)/den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.
    return min(point_segment_distance(p, x, y)
               for p, x, y in [(a,c,d), (b,c,d), (c,a,b), (d,a,b)])


def fk3(q):
    angle = np.cumsum(q)
    edges = np.array([.5, .4, .4])[:, None]*np.c_[np.cos(angle), np.sin(angle)]
    return np.vstack([np.zeros(2), np.cumsum(edges, axis=0)])


def main():
    out = Path(__file__).resolve().parents[2]/'_assets/3d_learning/day18'
    out.mkdir(parents=True, exist_ok=True)
    stride = 4
    center = np.array([77.2, 33.6])
    cell = np.floor(center/stride).astype(int)
    offset = center/stride-cell
    yy, xx = np.mgrid[:24, :30]
    heatmap = np.exp(-((xx-cell[0])**2+(yy-cell[1])**2)/2)
    row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    decoded = stride*(np.array([col,row])+offset)
    np.testing.assert_allclose(decoded, center)
    width, height = 12., 18.  # here defined in original-image pixels
    box = np.r_[decoded-[width,height]/np.array([2.,2.]),
                decoded+[width,height]/np.array([2.,2.])]
    def to_base(pixel):
        pcam = np.r_[(pixel-[60,48])/80, 1.]
        return np.diag([1,-1,-1])@pcam+[.4,0,1.2]
    reference = to_base(center)
    cases = []
    for name,pix in [('correct',decoded), ('offset_omitted',stride*np.array([col,row])),
                     ('stride_omitted',np.array([col,row])+offset)]:
        cases.append({'case':name, 'pixel':pix.tolist(), 'base_point_m':to_base(pix).tolist(),
                      'error_m':float(np.linalg.norm(to_base(pix)-reference))})
    assert np.isclose(cases[1]['error_m'], .025)
    radii = .025
    poses = [np.array([0., .7, -.2]), np.array([0., 2.5, 2.5])]
    collision = []
    for q in poses:
        nodes = fk3(q)
        clearance = segment_distance(*nodes[:2], *nodes[2:])-2*radii
        collision.append({'q_rad':q.tolist(), 'nodes_m':nodes.tolist(),
                          'link1_link3_clearance_m':clearance,
                          'self_collision':clearance <= 0,
                          'environment_collision':False})
    assert not collision[0]['self_collision'] and collision[1]['self_collision']
    report = {'metadata':{'date':'2026-09-22','python':platform.python_version(),
                          'numpy':np.__version__,'matplotlib':matplotlib.__version__,
                          'device':'CPU','seed':'none; deterministic geometry',
                          'training_steps':0,'robot_interactions':0},
              'center':{'stride':stride,'head_shape_hw':[24,30], 'cell_xy':cell.tolist(),
                        'offset':offset.tolist(),'decoded_box_xyxy_px':box.tolist(),'cases':cases},
              'collision':{'link_radius_m':radii,'obstacles':[],
                           'checked_pair':['link1','link3'],'poses':collision,
                           'scope':'Nonadjacent pair only; planar teaching model, no robot mesh'},
              'depth_assumption':'Known optical Z=1 m at the chosen ray; not inferred from the 2D box'}
    (out/'center-self-collision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.5),constrained_layout=True)
    axes[0].imshow(heatmap,origin='upper',extent=(0,120,96,0),cmap='Blues')
    axes[0].scatter(*center,color='#dc2626',label='Decoded center')
    axes[0].add_patch(Rectangle(box[:2],width,height,fill=False,color='#dc2626'))
    axes[0].set(title='Synthetic center head + offset + size',xlabel='Image u (px)',ylabel='Image v (px)')
    axes[0].legend(fontsize=8)
    for ax,item,name in zip(axes[1:],collision,['Separated nonadjacent links','Self collision, empty environment']):
        n=np.array(item['nodes_m'])
        for i,color in enumerate(['#2563eb','#94a3b8','#dc2626']):
            ax.plot(n[i:i+2,0],n[i:i+2,1],'o-',lw=6,color=color,label=f'Link {i+1}')
        ax.set(title=name,xlabel='Base x (m)',ylabel='Base y (m)',xlim=(-.1,1.25),ylim=(-.3,.75))
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('D18 | A detection center needs units; a free environment does not rule out self collision')
    fig.savefig(out/'center-self-collision.png',dpi=160)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
