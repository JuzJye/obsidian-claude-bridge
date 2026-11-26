"""D45: metric scale ambiguity, depth constraints and SO(3); CPU NumPy."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project(q):
    return q[:, :2] / q[:, 2:] * 600 + [320, 240]


def solve_scale_translation(canonical_rotated, pixels, depth, sigma_z):
    # Algebraic cross-multiplied projection residual; NOT a nonlinear PnP solver.
    x = canonical_rotated
    uv = (pixels - [320, 240]) / 600
    a = np.zeros((2 * len(x), 4))
    a[::2, 0] = x[:, 0] - uv[:, 0] * x[:, 2]
    a[1::2, 0] = x[:, 1] - uv[:, 1] * x[:, 2]
    a[::2, 1], a[1::2, 2] = 1, 1
    a[::2, 3], a[1::2, 3] = -uv[:, 0], -uv[:, 1]
    b = np.zeros(2 * len(x))
    # Approximate ray residual scale at nominal range .8 m, image noise .4 px.
    sigma_ray = .8 * .4 / 600
    az = np.c_[x[:, 2], np.zeros((len(x), 2)), np.ones(len(x))]
    matrix = np.r_[a / sigma_ray, az / sigma_z]
    target = np.r_[b, depth / sigma_z]
    solution = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return solution, np.linalg.svd(a, compute_uv=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-bias-mm", type=float, default=30.)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "_assets/3d_learning/day45")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(45)
    canonical = rng.uniform(-.5, .5, (80, 3))
    a = .45
    rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    x = canonical @ rotation.T
    scale, trans = .12, np.array([.03, -.01, .8])
    truth = scale*x + trans
    pixels = project(truth)
    noisy_pixels = pixels + rng.normal(0, .4, pixels.shape)
    noisy_depth = truth[:, 2] + rng.normal(0, .002, len(x))
    records = {}
    for name, depth in [("depth_correct", noisy_depth), ("depth_biased", noisy_depth + args.depth_bias_mm/1000)]:
        start = time.perf_counter()
        sol, sv = solve_scale_translation(x, noisy_pixels, depth, .002)
        q = sol[0]*x + sol[1:]
        records[name] = dict(scale_m=float(sol[0]), translation_m=sol[1:].tolist(),
            translation_error_mm=float(np.linalg.norm(sol[1:]-trans)*1000), scale_error_mm=float(abs(sol[0]-scale)*1000),
            clean_reprojection_rms_px=float(np.sqrt(np.mean(np.sum((project(q)-pixels)**2, axis=1)))),
            point_error_mm=float(np.linalg.norm(q-truth, axis=1).mean()*1000), solve_ms=(time.perf_counter()-start)*1000)
    alpha = np.linspace(.5, 1.5, 101)
    pixel_error = [np.sqrt(np.mean((project(k*truth)-pixels)**2)) for k in alpha]
    depth_error = [np.sqrt(np.mean((k*truth[:, 2]-truth[:, 2])**2))*1000 for k in alpha]
    # A coordinate network might produce a non-rotation: inspect and project.
    raw = rotation @ np.array([[1.10, .08, 0], [0, .92, .03], [0, 0, 1.05]])
    u, _, vt = np.linalg.svd(raw)
    corrected = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    report = dict(seed=45, timestamp="synthetic t=0", units="canonical unitless, metric m, images px", frame="object to camera", network_train_steps=0, robot_interactions=0, data_points=80,
        assumptions="Known rotation and correspondences; uniform scale unknown; algebraic weighted least squares, not full 6D PnP or NOCS reproduction.",
        nominal_scale_m=scale, nominal_translation_m=trans.tolist(), depth_bias_mm=args.depth_bias_mm, image_noise_px=.4, depth_noise_m=.002,
        environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__), results=records,
        scale_ambiguity_max_pixel_coordinate_rmse=float(max(pixel_error)), noisy_projection_singular_values=sv.tolist(),
        rotation_constraints=dict(raw_orthogonality_error=float(np.linalg.norm(raw.T@raw-np.eye(3))), raw_det=float(np.linalg.det(raw)), projected_orthogonality_error=float(np.linalg.norm(corrected.T@corrected-np.eye(3))), projected_det=float(np.linalg.det(corrected)), projected_angle_error_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(corrected@rotation.T)-1)/2, -1, 1))))))
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(alpha, pixel_error)
    ax[0].set(xlabel="common scale multiplier alpha", ylabel="image coordinate RMSE (px)", title="Scaling size and range leaves image unchanged")
    ax[1].plot(alpha, depth_error, color="tab:orange")
    ax[1].set(xlabel="common scale multiplier alpha", ylabel="depth RMSE (mm)", title="A calibrated metric sensor breaks ambiguity")
    fig.savefig(args.out_dir / "scale-ambiguity.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].bar(list(records), [v["clean_reprojection_rms_px"] for v in records.values()])
    ax[0].set(ylabel="clean image point RMS (px)", title="Both can look good in the image")
    ax[1].bar(list(records), [v["translation_error_mm"] for v in records.values()], color="tab:red")
    ax[1].set(ylabel="translation error (mm)", title="Depth bias transfers to robot geometry")
    fig.savefig(args.out_dir / "biased-constraint.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
