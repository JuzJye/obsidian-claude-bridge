"""Day50: metric depth, cross-camera registration and calibration failure injection."""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, W = 128, 192
KD = np.array([[150., 0, 95.5], [0, 150., 63.5], [0, 0, 1]])
KC = np.array([[180., 0, 95.5], [0, 180., 63.5], [0, 0, 1]])
T_CD = np.array([0.035, 0., 0.])  # p_color = p_depth + T_CD, parallel optical axes


def render(k, camera_center):
    v, u = np.indices((H, W))
    rays = np.stack([(u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1], np.ones_like(u)], -1)
    front = camera_center + .6*rays
    fg = (np.abs(front[..., 0]) < .10) & (np.abs(front[..., 1]) < .12)
    depth = np.where(fg, .6, 1.2)
    points = camera_center + depth[..., None]*rays
    checker = ((np.floor((points[..., 0]+1)*25) + np.floor((points[..., 1]+1)*25)) % 2)
    rgb = np.stack([.2+.5*fg, .25+.45*checker, .8-.55*fg], -1)
    return depth, rgb, fg


def deproject(depth, k):
    v, u = np.indices(depth.shape)
    return np.stack([(u-k[0, 2])*depth/k[0, 0],
                     (v-k[1, 2])*depth/k[1, 1], depth], -1)


def project(points, k):
    z = points[..., 2]
    return np.stack([k[0, 0]*points[..., 0]/z+k[0, 2],
                     k[1, 1]*points[..., 1]/z+k[1, 2]], -1)


def register(points_d, valid, translation):
    pc = points_d[valid] + translation
    uv_float = project(pc, KC)
    uv = np.rint(uv_float).astype(int)
    inside = (pc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    pc, uv = pc[inside], uv[inside]
    # Explicit custom output: Z in the COLOR optical frame, not a copied raw uint16.
    aligned = np.full(H*W, np.inf)
    np.minimum.at(aligned, uv[:, 1]*W+uv[:, 0], pc[:, 2])
    aligned = aligned.reshape(H, W)
    aligned[~np.isfinite(aligned)] = np.nan
    return aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--baseline-error-mm", type=float, default=0)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day50/results.json")
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    z_d, _, fg_d = render(KD, np.zeros(3))
    # Camera centre C in D is -t_CD because R_CD is identity.
    z_c, color, fg_c = render(KC, -T_CD)
    scale = .001  # metres per raw unit; actual device value must be queried
    raw = np.rint(z_d/scale).astype(np.uint16)
    missing = rng.random((H, W)) < .04
    raw[missing] = 0
    valid = raw > 0
    metric = raw.astype(np.float64)*scale
    points = deproject(metric, KD)
    estimated_t = T_CD + [args.baseline_error_mm/1000, 0, 0]
    registered = register(points, valid, estimated_t)
    wrong_registered = register(points, valid, np.zeros(3))
    overlap = np.isfinite(registered)
    foreground = valid & fg_d
    correct_uv = project(points[foreground]+T_CD, KC)
    bad_uv = project(points[foreground]+estimated_t, KC)
    naive_uv = np.column_stack(np.nonzero(foreground)[::-1])
    # Same-camera mathematical round trip: useful code check, not external calibration evidence.
    original_uv = np.column_stack(np.nonzero(valid)[::-1])
    roundtrip = project(points[valid], KD)
    assert np.max(np.abs(roundtrip-original_uv)) < 1e-10
    assert np.all(metric[valid] > 0) and not np.any(valid & (raw == 0))
    # Crop and resize with pixel-centre convention u' = s*(u-x0+0.5)-0.5.
    uv = np.array([[110., 70.]])
    cropped = .5*(uv-[20, 10]+.5)-.5
    k_new = KD.copy(); k_new[0, 0] *= .5; k_new[1, 1] *= .5
    k_new[:2, 2] = .5*(KD[:2, 2]-[20, 10]+.5)-.5
    ray_old = np.r_[((uv[0]-KD[:2, 2])/KD.diagonal()[:2]), 1.]
    ray_new = np.r_[((cropped[0]-k_new[:2, 2])/k_new.diagonal()[:2]), 1.]
    assert np.allclose(ray_old, ray_new)
    bad_ray = np.r_[((cropped[0]-KD[:2, 2])/KD.diagonal()[:2]), 1.]
    report = dict(python=platform.python_version(), numpy=np.__version__,
        matplotlib=matplotlib.__version__, device="CPU", seed=args.seed,
        data="synthetic two-plane RGB-D with two calibrated parallel-axis cameras",
        frame="optical x right, y down, z forward; p_C = p_D + t_CD",
        unit="m and pixel", depth_scale_m_per_unit=scale,
        timestamp="static synthetic t=0; motion budget evaluated analytically",
        raw_shape=list(raw.shape), valid_depth_pixels=int(valid.sum()),
        aligned_valid_pixels=int(overlap.sum()), total_pixels=H*W,
        missing_fraction=float(missing.mean()), baseline_error_mm=args.baseline_error_mm,
        foreground_registration_error_px=float(np.linalg.norm(bad_uv-correct_uv, axis=1).mean()),
        naive_same_pixel_error_px=float(np.linalg.norm(naive_uv-correct_uv, axis=1).mean()),
        registered_depth_disagreement_fraction=float((np.abs(registered[overlap]-z_c[overlap])>.01).mean()),
        ignored_extrinsic_disagreement_fraction=float((np.abs(wrong_registered[np.isfinite(wrong_registered)]-
            z_c[np.isfinite(wrong_registered)])>.01).mean()),
        roundtrip_max_error_px=float(np.abs(roundtrip-original_uv).max()),
        crop_resize_correct_ray_error=float(np.linalg.norm(ray_old-ray_new)),
        crop_resize_old_K_point_error_mm=float(600*np.linalg.norm(ray_old-bad_ray)),
        depth_scale_10pct_error_at_600mm=60.,
        motion_error_mm_at_0_2mps_30ms=6.,
        training_samples=0, hardware_connections=0, real_robot_interactions=0,
        elapsed_s=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"), raw_depth=raw, color=color,
        valid=valid, K_depth=KD, K_color=KC, t_color_depth=T_CD, registered_z_color=registered,
        depth_scale=scale, timestamp_depth_s=0., timestamp_color_s=0.)
    print(json.dumps(report, indent=2))
    if args.output.stem != "results":
        return
    asset = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day50"
    asset.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    axes[0, 0].imshow(color); axes[0, 0].set_title("Color camera: texture and occlusion")
    for ax, img, title in [(axes[0, 1], np.where(valid, metric, np.nan), "Raw depth in depth pixels"),
            (axes[1, 0], registered, "Registered Z_color; white = unknown"),
            (axes[1, 1], wrong_registered, "Wrong: ignore camera translation")]:
        im = ax.imshow(img, vmin=.55, vmax=1.25, cmap="viridis"); ax.set_title(title)
    for ax in axes.flat: ax.set(xlabel="u (pixel)", ylabel="v (pixel)")
    fig.colorbar(im, cax=fig.add_axes([.91, .2, .02, .6]), label="optical depth (m)")
    fig.subplots_adjust(wspace=.3, hspace=.4, right=.86)
    fig.savefig(asset/"rgbd_registration.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    depths = np.linspace(.3, 2, 100)
    for error_mm in [1, 3, 5]:
        axes[0].plot(depths, KC[0, 0]*error_mm/1000/depths, label=f"baseline error {error_mm} mm")
    axes[0].set(xlabel="Z (m)", ylabel="pixel error", title="Extrinsic translation error: f * delta_t / Z")
    dt = np.arange(0, 81, 5)
    for speed in [.1, .2, .5]: axes[1].plot(dt, speed*dt, label=f"speed {speed} m/s")
    axes[1].set(xlabel="RGB-depth time offset (ms)", ylabel="position mismatch (mm)", title="Time mismatch: speed * delay")
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(asset/"calibration_budgets.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
