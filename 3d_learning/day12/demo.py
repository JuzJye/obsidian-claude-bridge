"""Day12: multi-view OpenCV calibration with known synthetic metric truth."""
from pathlib import Path
import argparse
import json
import platform
import cv2
import numpy as np


def project(obj, pose, k, distortion):
    return cv2.projectPoints(obj, pose[0], pose[1], k, distortion)[0].reshape(-1, 2)


def solve(obj, images, flags=0):
    objects = [obj.astype(np.float32) for _ in images]
    return cv2.calibrateCamera(objects, [x.astype(np.float32) for x in images],
        (640,480), None, None, flags=flags | cv2.CALIB_FIX_K3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day12')
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    obj = np.zeros((9*6,3), np.float32)
    obj[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*.025
    obj[:,:2] -= obj[:,:2].mean(0)
    k = np.array([[620.,0,320],[0,610,240],[0,0,1]])
    distortion = np.array([-.18,.04,.001,-.0008,0.])
    poses = [(rng.uniform([-.45,-.45,-.3],[.45,.45,.3]),
              rng.uniform([-.13,-.09,.48],[.13,.09,.9])) for _ in range(24)]
    clean = np.array([project(obj, pose, k, distortion) for pose in poses])
    assert np.all((clean[...,0]>0)&(clean[...,0]<640)&(clean[...,1]>0)&(clean[...,1]<480))
    standard_noise = rng.normal(size=clean.shape)
    records, arrays = {}, dict(clean=clean, object_points=obj, true_K=k)
    for sigma in (.2, 1.):
        observed = clean + sigma*standard_noise
        rms, estimated_k, estimated_d, rv, tv = solve(obj, observed[:18])
        held_known, held_fitted = [], []
        for index in range(18,24):
            pred = project(obj, poses[index], estimated_k, estimated_d)
            held_known.append(np.mean(np.sum((pred-clean[index])**2,axis=1)))
            ok, r, t = cv2.solvePnP(obj, observed[index], estimated_k, estimated_d)
            assert ok
            fit = project(obj, (r,t), estimated_k, estimated_d)
            held_fitted.append(np.mean(np.sum((fit-clean[index])**2,axis=1)))
        records[f'noise_{sigma}'] = dict(training_rms_px=float(rms), K=estimated_k.tolist(),
            distortion=estimated_d.ravel().tolist(),
            focal_relative_error=float(np.linalg.norm(estimated_k.diagonal()[:2]-k.diagonal()[:2])/np.linalg.norm(k.diagonal()[:2])),
            heldout_fixed_truth_pose_rms_px=float(np.sqrt(np.mean(held_known))),
            heldout_fitted_pose_rms_px=float(np.sqrt(np.mean(held_fitted))))
        arrays[f'K_noise_{sigma}'] = estimated_k
        arrays[f'D_noise_{sigma}'] = estimated_d
        if sigma == .2:
            base = (rms, estimated_k, estimated_d, rv, tv)
            arrays['observed'] = observed
    # Same image points, only assumed board scale changes by 5%.
    scaled = solve(obj*1.05, arrays['observed'][:18])
    ratios = [np.linalg.norm(b)/np.linalg.norm(a) for a,b in zip(base[4], scaled[4])]
    records['board_scale_1.05'] = dict(training_rms_px=float(scaled[0]),
        translation_norm_ratio_mean=float(np.mean(ratios)),
        focal_relative_change=float(np.linalg.norm(scaled[1]-base[1])/np.linalg.norm(base[1])))
    # Omit distortion while optimizing the same unknowns other than fixed zero distortion.
    omitted = solve(obj, arrays['observed'][:18], cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2)
    records['zero_distortion_model'] = dict(training_rms_px=float(omitted[0]), K=omitted[1].tolist())
    full_errors, omit_errors = [], []
    for i in range(18):
        full_errors.append(project(obj,(base[3][i],base[4][i]),base[1],base[2])-arrays['observed'][i])
        omit_errors.append(project(obj,(omitted[3][i],omitted[4][i]),omitted[1],omitted[2])-arrays['observed'][i])
    arrays['full_residuals'] = np.array(full_errors); arrays['omitted_residuals'] = np.array(omit_errors)
    # Independently known depth and ray; isolate focal error / extrinsic rotation.
    points = np.array([[.0,0,.5],[.15,.10,.5],[.15,.10,1.0],[.3,.2,1.0]])
    pixel = project(points,(np.zeros(3),np.zeros(3)),k,np.zeros(5))
    wrong_k = k.copy(); wrong_k[0,0] *= 1.05; wrong_k[1,1] *= 1.05
    ray = cv2.undistortPoints(pixel.reshape(-1,1,2),wrong_k,np.zeros(5)).reshape(-1,2)
    reconstructed = np.column_stack((ray*points[:,2,None],points[:,2]))
    rotation,_ = cv2.Rodrigues(np.array([0,np.deg2rad(1),0]))
    rotated = points@rotation.T
    records['workspace'] = dict(points_m=points.tolist(),
        focal_plus5_error_m=np.linalg.norm(reconstructed-points,axis=1).tolist(),
        extrinsic_y_axis1deg_error_m=np.linalg.norm(rotated-points,axis=1).tolist())
    # A scalar idealized feedback example: residual measurement bias cannot be corrected away.
    histories = []
    for bias in (0., .01):
        error = .08; values = [error]
        for _ in range(15):
            error -= .5*(error+bias); values.append(error)
        histories.append(values)
    arrays['feedback_error'] = np.array(histories)
    report = dict(environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__),
        seed=args.seed, board=dict(inner_corners=[9,6], square_m=.025, views_train=18,views_holdout=6),
        true_K=k.tolist(), true_distortion=distortion.tolist(), results=records,
        feedback_final_error_m=[float(v[-1]) for v in histories],
        scope='synthetic corner observations, real calibrateCamera optimization; no detector/hand-eye/robot validation')
    assert base[0] < .35
    assert abs(np.mean(ratios)-1.05) < 1e-4
    assert records['noise_1.0']['training_rms_px'] > records['noise_0.2']['training_rms_px']
    assert omitted[0] > base[0]
    (args.out_dir/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.out_dir/'arrays.npz',**arrays)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
