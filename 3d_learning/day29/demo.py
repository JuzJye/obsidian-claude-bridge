"""Day29: 1D contact control and a synthetic tactile field, CPU only.

The ideal inner servo, linear wall and virtual mass are teaching assumptions.
No robot, measured tactile data, trained model or hardware safety validation.
"""
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def impedance(stiffness, wall_stiffness=2000., damping=90., dt=.0005):
    t = np.arange(0., 1.5+dt/2, dt)
    x, v = -.003, 0.
    trace = []
    for time in t:
        force = wall_stiffness*max(x, 0.)
        trace.append([time, x, v, force])
        acceleration = (stiffness*(.005-x)-damping*v-force)/1.
        v += dt*acceleration
        x += dt*v
    return np.array(trace)


def admittance(damping=90., wall_stiffness=2000., wall=0., dt=.0005):
    t = np.arange(0., 2.+dt/2, dt)
    x, v = -.003, 0.
    trace = []
    for time in t:
        force = wall_stiffness*max(x-wall, 0.)
        trace.append([time, x, v, force])
        acceleration = (5.-force-damping*v)/1.
        v = np.clip(v+dt*acceleration, -.05, .05)
        x += dt*v
    return np.array(trace)


def force_iteration(gain, steps=20, dt=.01, wall_stiffness=2000.):
    # Signed bilateral spring, initialized and remaining in contact in stable case.
    x = .001
    force = []
    for _ in range(steps+1):
        force.append(wall_stiffness*x)
        x += dt*gain*(5.-wall_stiffness*x)
    return np.array(force)


def main():
    out = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day29"
    out.mkdir(parents=True, exist_ok=True)
    imp = {k: impedance(k) for k in (200., 2000.)}
    adm = {b: admittance(b) for b in (20., 90.)}
    shifted = admittance(90., wall=.003)
    static = []
    for wall_k in (500., 2000., 8000.):
        static.append(dict(wall_stiffness_N_per_m=wall_k,
            ideal_position_force_N=wall_k*.005,
            impedance_K200_force_N=wall_k*200/(wall_k+200)*.005))
    summaries = {}
    for k, tr in imp.items():
        expected = k*2000/(k+2000)*.005
        assert abs(tr[-1, 3]-expected) < .001
        summaries[f"impedance_K{k:g}"] = dict(peak_N=float(tr[:, 3].max()),
            final_N=float(tr[-1, 3]), theory_N=expected)
    for b, tr in adm.items():
        assert abs(tr[-1, 3]-5) < .001
        summaries[f"admittance_B{b:g}"] = dict(peak_N=float(tr[:, 3].max()),
            final_N=float(tr[-1, 3]))
    assert abs(shifted[-1, 3]-5) < .001
    assert abs(shifted[-1, 1]-.0055) < 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for k, tr in imp.items():
        axes[0].plot(tr[:, 0], tr[:, 3], label=f"K={k:g} N/m")
    axes[0].set(xlabel="Time [s]", ylabel="Contact force [N]",
                title="Same reference, different impedance")
    axes[0].legend()
    wall_ks = [r["wall_stiffness_N_per_m"] for r in static]
    axes[1].plot(wall_ks, [r["ideal_position_force_N"] for r in static], "o-", label="Ideal position: 5 mm")
    axes[1].plot(wall_ks, [r["impedance_K200_force_N"] for r in static], "s-", label="Impedance: K=200 N/m")
    axes[1].set(xlabel="Wall stiffness [N/m]", ylabel="Static force [N]",
                title="One variable: environment stiffness")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out/"contact_stiffness.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for b, tr in adm.items():
        axes[0].plot(tr[:, 0], tr[:, 3], label=f"B={b:g} Ns/m")
    axes[0].axhline(5, color="black", ls="--", label="Target: 5 N")
    axes[0].set(xlabel="Time [s]", ylabel="Force [N]", title="Admittance: same target, damping changes")
    axes[0].legend()
    base = adm[90.]
    axes[1].plot(base[:, 0], base[:, 1]*1000, label="Wall at 0 mm")
    axes[1].plot(shifted[:, 0], shifted[:, 1]*1000, label="Wall at 3 mm")
    axes[1].axhline(2.5, color="C0", ls=":")
    axes[1].axhline(5.5, color="C1", ls=":")
    axes[1].set(xlabel="Time [s]", ylabel="Position [mm]", title="Same force; contact location changes")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out/"admittance_response.png", dpi=170)
    plt.close(fig)

    # Synthetic deformation, not a force reconstruction from real RGB.
    x = np.linspace(-10., 10., 61)
    y = np.linspace(-8., 8., 49)
    xx, yy = np.meshgrid(x, y)
    height = .8*np.exp(-((xx/4)**2+(yy/3)**2)/2)
    dhdy, dhdx = np.gradient(height, y, x)
    normals = np.stack([-dhdx, -dhdy, np.ones_like(height)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    marker_displacement = .6*np.exp(-((xx/6)**2+(yy/5)**2)/2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    im = axes[0].imshow(height, origin="lower", extent=[-10, 10, -8, 8], cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="Deformation [mm]")
    axes[0].set(title="Synthetic contact geometry", xlabel="Gel x [mm]", ylabel="Gel y [mm]")
    sel = np.s_[::4, ::4]
    axes[1].quiver(xx[sel], yy[sel], marker_displacement[sel], np.zeros_like(xx[sel]),
                   angles="xy", scale_units="xy", scale=1, width=.005)
    axes[1].set(title="Synthetic shear displacement (1:1)", xlabel="Gel x [mm]", ylabel="Gel y [mm]", aspect="equal")
    fig.savefig(out/"tactile_fields.png", dpi=170)
    plt.close(fig)
    good, bad = force_iteration(.03), force_iteration(.12)
    assert abs(good[-1]-5) < 1e-5
    assert abs(bad[-1]-5) > 1000
    report = dict(scope="CPU teaching simulation; no robot or GelSight inference",
        environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        static_force=static, responses=summaries,
        shifted_wall_final_position_mm=float(shifted[-1, 1]*1000),
        discrete_force=dict(stable_multiplier=.4, unstable_multiplier=-1.4,
            stable_final_N=float(good[-1]), unstable_final_signed_N=float(bad[-1]),
            caveat="Unstable signed linear spring loses unilateral-contact validity"),
        tactile_shapes=dict(height=list(height.shape), normals=list(normals.shape)))
    (out/"results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    np.savetxt(out/"admittance_B90.csv", base, delimiter=",", header="time_s,x_m,v_m_per_s,force_N", comments="")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
