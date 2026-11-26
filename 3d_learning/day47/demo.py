"""D47: parallel-jaw contact candidates on an analytic elliptical cylinder."""
from pathlib import Path
import argparse, json, platform, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])


def transform(r, t):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=.4)
    ap.add_argument("--max-width-mm", type=float, default=80.)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2]/"_assets/3d_learning/day47")
    args = ap.parse_args()
    if args.mu <= 0 or args.max_width_mm <= 0:
        raise ValueError("mu and width must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(47)
    radii = np.array([.028, .04])
    heights = [.025, .05, .075]
    max_width, margin = args.max_width_mm/1000, .004
    mass, force_per_finger, gravity = .25, 4., 9.81
    load_margin = 2*args.mu*force_per_finger-mass*gravity
    candidates = []
    start = time.perf_counter()
    for j, angle in enumerate(np.linspace(0, np.pi, 18, endpoint=False)):
        xy = radii*np.array([np.cos(angle), np.sin(angle)])
        p = np.r_[xy, 0.]
        closing = p/np.linalg.norm(p)
        normal = np.r_[xy/radii**2, 0.]
        normal /= np.linalg.norm(normal)
        beta = float(np.arccos(np.clip(normal@closing, -1, 1)))
        required_mu = float(np.tan(beta))
        capacity = 2*force_per_finger*np.sqrt(max(0., args.mu**2-required_mu**2))
        coupled_margin = capacity-mass*gravity
        contact_width = float(2*np.linalg.norm(p))
        width = contact_width+margin
        approach = np.array([0., 0., -1.])
        rotation = np.column_stack([approach, closing, np.cross(approach, closing)])
        assert np.allclose(rotation.T@rotation, np.eye(3)) and np.linalg.det(rotation) > .999
        for h in heights:
            center = np.array([0., 0., h])
            # Two opposing contact normal errors; this tests an analytic predicate only.
            beta_noisy = np.abs(beta+rng.normal(0, np.deg2rad(5), (300, 2)))
            width_noisy = width+rng.normal(0, .002, 300)
            noisy_mu = np.tan(beta_noisy).max(axis=1)
            noisy_capacity = 2*force_per_finger*np.sqrt(np.maximum(0., args.mu**2-noisy_mu**2))
            pass_samples = ((noisy_mu <= args.mu) & (width_noisy <= max_width) & (noisy_capacity >= mass*gravity))
            width_ok, friction_ok = width <= max_width, required_mu <= args.mu
            valid = bool(width_ok and friction_ok and coupled_margin >= 0)
            # A heuristic score, not an execution success probability.
            score = max(0, 1-required_mu/args.mu)*max(0, 1-width/max_width)*np.exp(-((h-.05)/.03)**2) if valid else 0.
            candidates.append(dict(id=f"a{j}-h{h:.3f}", angle_deg=float(np.rad2deg(angle)), center_m=center.tolist(), rotation=rotation.tolist(), width_m=width, contact_width_m=contact_width,
                minimum_mu=required_mu, beta_deg=float(np.rad2deg(beta)), width_ok=bool(width_ok), friction_ok=bool(friction_ok), load_ok=bool(coupled_margin>=0), coupled_load_margin_N=float(coupled_margin), geometric_candidate=valid, score=float(score),
                perturbation_passes=int(pass_samples.sum()), perturbation_trials=300))
    best = max(candidates, key=lambda c:c["score"])
    tog = transform(np.array(best["rotation"]), np.array(best["center_m"]))
    tbc = transform(rz(np.pi/2), [.4, -.2, .1])
    tco = transform(rz(np.pi/6), [.1, -.03, .6])
    tbg = tbc@tco@tog
    tgf = transform(np.eye(3), [-.10, 0, 0])
    tbf = tbg@tgf
    pregrasp = tbg[:3, 3]-.08*tbg[:3, 0]
    elapsed = (time.perf_counter()-start)*1000
    report = dict(seed=47, environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__), units="m, N, kg, radians internally", timestamp="synthetic t=0", frame_convention="T_AB maps B to A; gripper x=approach, y=closing, z=x cross y", geometry="analytic elliptical cylinder, radii 28/40 mm; hard point contacts with Coulomb friction",
        mu=args.mu, max_width_mm=args.max_width_mm, opening_margin_mm=4, mass_kg=mass, force_per_finger_N=force_per_finger, load_margin_N=load_margin,
        candidate_count=len(candidates), predicates_pass=sum(c["geometric_candidate"] for c in candidates), robot_attempts=0, optimization_training_steps=0, generate_and_perturb_ms=elapsed,
        chosen_candidate=best if best["score"]>0 else None, transform_example_even_if_rejected=dict(T_BC=tbc.tolist(),T_CO=tco.tolist(),T_OG=tog.tolist(),T_BG=tbg.tolist(),T_BF=tbf.tolist(),pregrasp_B_m=pregrasp.tolist()),
        scope="Analytic candidate predicates, not full 6D force closure, collision/IK, grasp policy, or robot success; scores are not calibrated probabilities.")
    (args.out_dir/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir/"candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    mid = [c for c in candidates if c["center_m"][2] == .05]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    theta = np.linspace(0, 2*np.pi, 300)
    ax[0].plot(28*np.cos(theta), 40*np.sin(theta), color="black")
    for c in mid:
        y = np.array(c["rotation"])[:, 1]
        half = c["contact_width_m"]/2*y*1000
        ax[0].plot([-half[0],half[0]],[-half[1],half[1]], color="tab:green" if c["geometric_candidate"] else "tab:red", alpha=.45)
    ax[0].set(xlabel="object x (mm)", ylabel="object y (mm)", title="Contact lines: predicate pass (green), reject (red)")
    ax[0].set_aspect("equal")
    angles = [c["angle_deg"] for c in mid]
    ax[1].plot(angles, [c["minimum_mu"] for c in mid], label="required mu")
    ax[1].axhline(args.mu, color="black", linestyle="--", label="assumed mu")
    ax[1].set(xlabel="ellipse parameter angle (deg)", ylabel="friction coefficient", title="Opposite contacts need not align with normals")
    ax[1].legend()
    fig.savefig(args.out_dir/"contact-geometry.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].plot(angles, [c["width_m"]*1000 for c in mid], marker="o")
    ax[0].axhline(max_width*1000, color="red", linestyle="--", label="max opening")
    ax[0].set(xlabel="ellipse parameter angle (deg)", ylabel="required opening (mm)", title="Object span + insertion clearance")
    ax[0].legend()
    ax[1].bar(angles, [c["perturbation_passes"]/300 for c in mid], width=7)
    ax[1].set(xlabel="ellipse parameter angle (deg)", ylabel="fraction passing perturbed analytic test", ylim=(0,1.05), title="300 perturbed predicates per candidate; not robot trials")
    fig.savefig(args.out_dir/"width-and-uncertainty.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
