"""Day 8: metric projection, coordinate chains, and unit contracts. CPU only."""
from pathlib import Path
import argparse
import json
import platform
import numpy as np


def transform(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inverse(T):
    R, t = T[:3, :3], T[:3, 3]
    return transform(R.T, -R.T @ t)


def apply(T, xyz):
    return np.asarray(xyz) @ T[:3, :3].T + T[:3, 3]


def project(K, xyz):
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("Projection requires positive optical z")
    h = xyz @ K.T
    return h[:, :2] / h[:, 2:3]


def unproject(K, uv, z):
    z = np.asarray(z)
    if np.any(~np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("Depth must be finite positive optical z")
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    return rays * z[:, None]


def main():
    default = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day8"
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=default)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    # Optical z points downward in base; proper rotation det(R)=+1.
    T_be = transform(np.diag([1., -1., -1.]), [.3, .1, 1.])
    T_ec = transform(np.eye(3), [.05, 0., .02])
    T_et = transform(np.eye(3), [0., 0., .12])
    T_bc, T_bt = T_be @ T_ec, T_be @ T_et
    pc = np.array([[.1, .05, .5], [-.08, .04, .6], [.03, -.06, .4]])
    uv = project(K, pc)
    back = unproject(K, uv, pc[:, 2])
    pb = apply(T_bc, pc)
    closed = apply(T_bc, apply(inverse(T_bc), pb))
    # Controlled ablation: identical pixels/K/pose, change only depth units.
    raw_mm = pc[:, 2] * 1000.
    correct = apply(T_bc, unproject(K, uv, raw_mm * .001))
    wrong = apply(T_bc, unproject(K, uv, raw_mm))
    # Range is not optical z: same ray, Euclidean distance differs.
    ranges = np.linalg.norm(pc, axis=1)
    range_as_z = unproject(K, uv, ranges)
    ray = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    range_correct = ray / np.linalg.norm(ray, axis=1, keepdims=True) * ranges[:, None]
    delta_cam = np.array([0., 0., .01])
    delta_base = T_bc[:3, :3] @ delta_cam
    # Resize/crop affects K; pixel centers follow ideal continuous coordinates.
    scale = .5
    K_half = K.copy()
    K_half[:2] *= scale
    resized_back = unproject(K_half, uv * scale, pc[:, 2])
    stale_K_back = unproject(K, uv * scale, pc[:, 2])
    assert np.max(np.abs(back - pc)) < 1e-12
    assert np.max(np.abs(closed - pb)) < 1e-12
    assert np.max(np.abs(correct - pb)) < 1e-12
    assert np.max(np.abs(range_correct - pc)) < 1e-12
    assert np.max(np.abs(resized_back - pc)) < 1e-12
    assert np.isclose(np.linalg.det(T_bc[:3, :3]), 1.)
    result = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "units": {"xyz": "m", "pixels": "px", "rotation": "rad"},
        "T_base_camera": T_bc.tolist(), "T_base_tcp": T_bt.tolist(),
        "pixel_uv": uv.tolist(), "points_base_m": pb.tolist(),
        "roundtrip_max_error_m": float(np.max(np.linalg.norm(back-pc, axis=1))),
        "unit_error_m": np.linalg.norm(wrong-correct, axis=1).tolist(),
        "range_as_z_error_m": np.linalg.norm(range_as_z-pc, axis=1).tolist(),
        "stale_K_resize_error_m": np.linalg.norm(stale_K_back-pc, axis=1).tolist(),
        "camera_delta_m": delta_cam.tolist(), "base_delta_m": delta_base.tolist(),
        "scope": "synthetic pinhole geometry; no calibration fit or robot execution",
    }
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez(args.out_dir / "arrays.npz", K=K, pc=pc, pb=pb, uv=uv, T_bc=T_bc,
             T_be=T_be, T_bt=T_bt, correct=correct, wrong=wrong,
             range_error=np.linalg.norm(range_as_z-pc, axis=1),
             stale_K_error=np.linalg.norm(stale_K_back-pc, axis=1))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
