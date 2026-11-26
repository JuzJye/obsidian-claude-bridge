"""Day14 cumulative geometry audit and toy parameter-transfer experiment."""
from pathlib import Path
import argparse
import json
import platform
import cv2
import numpy as np


def triangulate(p1,p2,x1,x2):
    a=np.stack((x1[0]*p1[2]-p1[0],x1[1]*p1[2]-p1[1],x2[0]*p2[2]-p2[0],x2[1]*p2[2]-p2[1]))
    _,_,vh=np.linalg.svd(a); h=vh[-1]
    return h[:3]/h[3]


def fk(q):
    return np.array([.5*np.cos(q[0])+.4*np.cos(q.sum()),.5*np.sin(q[0])+.4*np.sin(q.sum())])


def jac(q):
    return np.array([[-.5*np.sin(q[0])-.4*np.sin(q.sum()),-.4*np.sin(q.sum())],
                     [.5*np.cos(q[0])+.4*np.cos(q.sum()),.4*np.cos(q.sum())]])


def optimize(x,y,initial,trainable,steps=80):
    w=initial.copy(); histories=[]
    for _ in range(steps):
        error=x@w-y
        histories.append(float(np.mean(error**2)))
        grad=2*x.T@error/len(y)
        w-=.12*grad*trainable
    return w,np.array(histories)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=14)
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day14')
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    k=np.array([[600.,0,320],[0,600,240],[0,0,1]])
    p1=k@np.column_stack((np.eye(3),np.zeros(3)))
    p2=k@np.column_stack((np.eye(3),np.array([-.06,0,0])))
    point=np.array([.12,.04,.8]); h=np.r_[point,1.]
    x1=p1@h; x1=x1[:2]/x1[2]; x2=p2@h; x2=x2[:2]/x2[2]
    recovered=triangulate(p1,p2,x1,x2)
    scale=1.05
    scaled_p2=k@np.column_stack((np.eye(3),np.array([-.06*scale,0,0])))
    wrong=triangulate(p1,scaled_p2,x1,x2)
    reproject=p1@np.r_[wrong,1]; reproject=reproject[:2]/reproject[2]
    noisy=rng.normal(0,.25,20000)
    depths=36./(45+noisy)
    q=np.array([.4,-.8]); eps=1e-6
    fd=np.column_stack([(fk(q+np.eye(2)[i]*eps)-fk(q-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    j=jac(q); singular=jac(np.zeros(2))
    delta=np.array([.002,-.001]); dq=np.linalg.solve(j,delta)
    linearization_error=np.linalg.norm((fk(q+dq)-fk(q))-delta)
    # Source pretraining is actually fitted, not a hand-written checkpoint.
    source_x=np.column_stack((rng.normal(size=(500,2)),np.ones(500)))
    source_y=source_x@np.array([1.2,-.6,.1])+rng.normal(0,.03,500)
    pretrained,prehistory=optimize(source_x,source_y,np.zeros(3),np.ones(3),100)
    target_x=np.column_stack((rng.normal(size=(12,2)),np.ones(12)))
    target_y=target_x@np.array([1.5,-.8,.3])+rng.normal(0,.03,12)
    test_x=np.column_stack((rng.normal(size=(1000,2)),np.ones(1000)))
    test_y=test_x@np.array([1.5,-.8,.3])
    cases={'zero_shot':(pretrained,np.zeros(3)), 'bias_only':(pretrained,np.array([0,0,1])),
           'full_finetune':(pretrained,np.ones(3)), 'scratch':(np.zeros(3),np.ones(3))}
    records,arrays={},dict(pretrain_loss=prehistory,depth_samples=depths)
    for name,(initial,mask) in cases.items():
        fitted,history=optimize(target_x,target_y,initial,mask)
        records[name]=dict(updated_parameters=int(mask.sum()),weights=fitted.tolist(),
                          target_test_rmse=float(np.sqrt(np.mean((test_x@fitted-test_y)**2))))
        arrays[f'loss_{name}']=history
    # Separate expressivity check: fitted source slopes and noise-free target labels.
    no_shift_y=target_x@np.r_[pretrained[:2],.3]
    frozen,_=optimize(target_x,no_shift_y,pretrained,np.array([0,0,1]))
    records['bias_only_no_slope_shift']=dict(target_test_rmse=float(np.sqrt(np.mean((test_x@frozen-test_x@np.r_[pretrained[:2],.3])**2))))
    report=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__),seed=args.seed,
        geometry=dict(triangulation_error_m=float(np.linalg.norm(recovered-point)),
            baseline_plus5_error_m=float(np.linalg.norm(wrong-point)), same_wrong_model_reprojection_px=float(np.linalg.norm(reproject-x1)),
            depth_sigma_empirical_m=float(depths.std()), depth_sigma_linear_m=float(.8**2/36*.25),
            jacobian_fd_max_error=float(np.max(np.abs(fd-j))), regular_singular_values=np.linalg.svd(j)[1].tolist(),
            straight_arm_singular_values=np.linalg.svd(singular)[1].tolist(), small_step_linearization_error_m=float(linearization_error)),
        transfer=records, pretrain_samples=500,target_train_samples=12,target_test_samples=1000,
        scope='geometry unit audit + fitted three-parameter linear transfer; no foundation model or robot training')
    assert report['geometry']['triangulation_error_m'] < 1e-10
    assert report['geometry']['same_wrong_model_reprojection_px'] < 1e-8
    assert report['geometry']['jacobian_fd_max_error'] < 1e-8
    assert abs(report['geometry']['depth_sigma_empirical_m']/report['geometry']['depth_sigma_linear_m']-1)<.05
    assert records['full_finetune']['target_test_rmse'] < records['bias_only']['target_test_rmse']
    (args.out_dir/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.out_dir/'arrays.npz',**arrays)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
