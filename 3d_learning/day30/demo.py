"""Day30: planar visual servo and force-budget slip correction, CPU only.

Two translational camera DOFs; synthetic force-informed tactile trigger.
Not a complete six-DOF servo, GelSight network, friction model or robot test.
"""
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def servo(depth_ratio=1., mode="ibvs", pose_bias=0., steps=100):
    dt, gain, depth, focal = .05, 2., .5, 600.
    point = np.array([.08, -.04])
    camera = np.zeros(2)
    trace = []
    for k in range(steps+1):
        feature = (point-camera)/depth
        trace.append([k*dt, *camera, np.linalg.norm(feature)*focal])
        if k == steps:
            break
        if mode == "ibvs":
            velocity = gain*depth*depth_ratio*feature
        else:
            velocity = gain*(point-camera+np.array([pose_bias, 0.]))
        camera += dt*velocity
    return np.array(trace)


def slip(use_tactile, max_force=12.):
    dt, mu, drag = .01, .4, 80.
    t = np.arange(0., 2.+dt/2, dt)
    normal, displacement, active = 6., 0., False
    delayed_displacement = []
    trace = []
    trigger_time = None
    for k, time in enumerate(t):
        load = 1.5 if time < .5 else 4.
        # Common camera trigger: >2 mm accumulated slip, 100 ms sensor delay.
        camera_d = delayed_displacement[max(0, k-10)] if k else 0.
        visual_trigger = camera_d > .002
        # Idealized calibrated force warning, not raw tactile image inference.
        tactile_trigger = use_tactile and load > .9*mu*normal
        if (visual_trigger or tactile_trigger) and not active:
            active, trigger_time = True, float(time)
        if active:
            normal = min(max_force, normal+12.*dt)
        velocity = max(load-mu*normal, 0.)/drag
        displacement += dt*velocity
        delayed_displacement.append(displacement)
        trace.append([time, normal, load, velocity, displacement])
    return np.array(trace), trigger_time


def point_interaction(x, y, z):
    return np.array([[-1/z, 0, x/z, x*y, -(1+x*x), y],
                     [0, -1/z, y/z, 1+y*y, -x*y, -x]])


def main():
    out = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day30"
    out.mkdir(parents=True, exist_ok=True)
    runs = {r: servo(r) for r in (.5, 1., 4.)}
    biased = servo(mode="pbvs", pose_bias=.01)
    unbiased = servo(mode="pbvs")
    reverse = servo(-1., steps=15)
    assert runs[1.][-1, 3] < .01
    assert abs(biased[-1, 3]-12) < .01
    assert reverse[-1, 3] > reverse[0, 3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ratio, tr in runs.items():
        axes[0].semilogy(tr[:, 0], np.maximum(tr[:, 3], 1e-9), label=f"Z_hat/Z={ratio:g}")
    axes[0].set(xlabel="Time [s]", ylabel="Image error [px]", title="IBVS: positive depth mismatch")
    axes[0].legend()
    for label, tr in [("PBVS: unbiased", unbiased), ("PBVS: +10 mm pose bias", biased), ("IBVS: reversed sign", reverse)]:
        axes[1].semilogy(tr[:, 0], np.maximum(tr[:, 3], 1e-9), label=label)
    axes[1].set(xlabel="Time [s]", ylabel="Image error [px]", title="Controlled bias / sign counterexamples")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out/"visual_servo.png", dpi=170)
    plt.close(fig)

    tactile, tt = slip(True)
    vision, tv = slip(False)
    limited, tl = slip(True, max_force=8.)
    assert tactile[-1, 4] < vision[-1, 4]
    assert limited[-1, 3] > 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for label, tr in [("Vision only", vision), ("Vision + force warning", tactile), ("Same warning, 8 N cap", limited)]:
        axes[0].plot(tr[:, 0], tr[:, 1], label=label)
        axes[1].plot(tr[:, 0], tr[:, 4]*1000, label=label)
    axes[0].axhline(10, ls="--", color="black", label="Required at mu=0.4")
    axes[0].set(xlabel="Time [s]", ylabel="Total normal load [N]", title="Bounded corrective squeeze")
    axes[1].set(xlabel="Time [s]", ylabel="Accumulated slip [mm]", title="Early trigger helps; force cap still binds")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(out/"slip_feedback.png", dpi=170)
    plt.close(fig)

    # Four normalized image points provide 8 rows for 6 camera-twist columns.
    points = np.array([[-.1, -.1, .5], [.1, -.1, .5], [.1, .1, .5], [-.1, .1, .5]])
    interaction = np.vstack([point_interaction(*p) for p in points])
    singular = np.linalg.svd(interaction, compute_uv=False)
    assert interaction.shape == (8, 6)
    assert np.linalg.matrix_rank(interaction) == 6
    # Velocity check in the restricted x/y translation setting.
    p = np.array([.08, -.04, .5])
    v = np.array([.01, -.02, 0., 0., 0., 0.])
    epsilon = 1e-6
    old = p[:2]/p[2]
    moved = p-epsilon*v[:3]
    finite_difference = (moved[:2]/moved[2]-old)/epsilon
    analytic = point_interaction(*old, p[2])@v
    assert np.max(np.abs(finite_difference-analytic)) < 1e-8
    report = dict(scope="Planar camera + toy contact feedback; no learned slip detector",
        environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        servo_final_px={f"depth_ratio_{r:g}":float(tr[-1, 3]) for r, tr in runs.items()},
        pbvs_bias_final_px=float(biased[-1, 3]),
        reversed_sign_15_steps_px=float(reverse[-1, 3]),
        interaction_shape=list(interaction.shape), interaction_rank=int(np.linalg.matrix_rank(interaction)),
        interaction_singular_values=singular.tolist(), derivative_max_error=float(np.max(np.abs(finite_difference-analytic))),
        slip={name:dict(trigger_s=when, total_slip_mm=float(tr[-1, 4]*1000),
                       final_slip_mm_per_s=float(tr[-1, 3]*1000), final_normal_N=float(tr[-1, 1]))
              for name, tr, when in [("vision_only", vision, tv), ("fusion", tactile, tt), ("force_cap_8N", limited, tl)]})
    (out/"results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    np.savetxt(out/"slip_fusion.csv", tactile, delimiter=",",
               header="time_s,normal_N,tangential_N,slip_velocity_m_per_s,slip_m", comments="")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
