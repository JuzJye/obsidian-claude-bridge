"""D23 handwritten anchored BA: analytic Jacobians, IRLS, damped Schur step.
No scipy optimizer and no COLMAP. First two camera poses are known metric anchors.
T_cw maps world to camera; q=R X+t. Metres, pixels, radians. Seed 23.
"""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "_assets/3d_learning/day23"
F, CX, CY = 650., 320., 240.


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_so3(w):
    angle = np.linalg.norm(w)
    W = skew(w)
    if angle < 1e-8:
        return np.eye(3) + W + .5 * W @ W
    return np.eye(3) + np.sin(angle)/angle*W + (1-np.cos(angle))/angle**2*W@W


def project(q):
    return np.array([F*q[0]/q[2]+CX, F*q[1]/q[2]+CY])


def jacobians(R, t, X):
    q = R @ X + t
    x, y, z = q
    P = np.array([[F/z, 0, -F*x/z**2], [0, F/z, -F*y/z**2]])
    # Local pose retraction: R'=Exp(dw)R, t'=Exp(dw)t+dv.
    # This is a first-order SE(3) retraction; dv is local camera translation.
    return P @ np.column_stack([-skew(q), np.eye(3)]), P @ R


def residuals(R, t, X, observations):
    return np.array([project(R[i]@X[j]+t[i])-uv for i, j, uv in observations])


def cost_and_weights(r, robust):
    norms = np.linalg.norm(r, axis=1)
    if not robust:
        return float(.5*np.sum(norms**2)), np.ones(len(r))
    delta = 1.
    costs = np.where(norms <= delta, .5*norms**2, delta*(norms-.5*delta))
    return float(np.sum(costs)), np.minimum(1., delta/np.maximum(norms, 1e-15))


def linearize(R, t, X, obs, robust):
    ncam = 6*(len(R)-2)
    J = np.zeros((2*len(obs), ncam+3*len(X)))
    r = residuals(R, t, X, obs)
    cost, weights = cost_and_weights(r, robust)
    for k, (i, j, _) in enumerate(obs):
        A, B = jacobians(R[i], t[i], X[j])
        if i >= 2:
            J[2*k:2*k+2, 6*(i-2):6*(i-1)] = A
        J[2*k:2*k+2, ncam+3*j:ncam+3*j+3] = B
    sqrtw = np.repeat(np.sqrt(weights), 2)
    return cost, J*sqrtw[:, None], r.ravel()*sqrtw


def schur_step(J, r, ncam, damping):
    H, g = J.T@J, J.T@r
    H += np.diag(damping*np.maximum(np.diag(H), 1.))
    U, W, V = H[:ncam, :ncam], H[:ncam, ncam:], H[ncam:, ncam:]
    gc, gp = g[:ncam], g[ncam:]
    # V is block diagonal: each observation touches exactly one 3D point.
    Vinv = np.zeros_like(V)
    for j in range(0, len(V), 3):
        Vinv[j:j+3, j:j+3] = np.linalg.solve(V[j:j+3, j:j+3], np.eye(3))
    dc = np.linalg.solve(U-W@Vinv@W.T, -gc+W@Vinv@gp)
    dp = -Vinv@(gp+W.T@dc)
    return np.r_[dc, dp], H, g


def apply_step(R, t, X, step):
    rr, tt = R.copy(), t.copy()
    ncam = 6*(len(R)-2)
    for i in range(2, len(R)):
        delta = step[6*(i-2):6*(i-1)]
        Q = exp_so3(delta[:3])
        rr[i], tt[i] = Q@R[i], Q@t[i]+delta[3:]
    return rr, tt, X+step[ncam:].reshape(-1, 3)


def optimize(R, t, X, obs, heldout, robust):
    R, t, X = R.copy(), t.copy(), X.copy()
    damping, history = 1e-3, []
    for iteration in range(45):
        cost, J, r = linearize(R, t, X, obs, robust)
        history.append([iteration, cost, np.sqrt(np.mean(residuals(R,t,X,heldout)**2))])
        step, _, _ = schur_step(J, r, 6*(len(R)-2), damping)
        nr, nt, nx = apply_step(R, t, X, step)
        positive = all((nr[i]@nx[j]+nt[i])[2] > .1 for i,j,_ in obs)
        newcost = cost_and_weights(residuals(nr, nt, nx, obs), robust)[0] if positive else np.inf
        if newcost < cost:
            R, t, X = nr, nt, nx
            damping = max(damping/3, 1e-9)
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            damping = min(damping*10, 1e10)
    history.append([iteration+1, cost_and_weights(residuals(R,t,X,obs),robust)[0],
                    np.sqrt(np.mean(residuals(R,t,X,heldout)**2))])
    return R, t, X, np.array(history)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)
    X = rng.uniform([-.9, -.7, 3.5], [.9, .7, 5.5], (32,3))
    C = np.array([[0.,0,0],[.7,0,0],[.25,.4,.05],[.95,.3,-.05]])
    R = np.array([exp_so3(np.array(w)) for w in [[0,0,0],[0,-.03,0],
                  [.03,-.015,0],[-.02,.02,.01]]])
    t = -np.einsum("nij,nj->ni", R, C)
    obs, heldout, outlier_flags = [], [], []
    for i in range(4):
        for j in range(len(X)):
            uv = project(R[i]@X[j]+t[i])
            if i >= 2 and j % 4 == i-2:
                # Clean withheld projections, NEVER passed to the optimizer.
                heldout.append((i,j,uv))
            else:
                bad = i >= 2 and j in [6,11,18,23,30]
                z = uv+rng.normal(0,.3,2)
                if bad:
                    z += np.array([18.,-12.])
                obs.append((i,j,z))
                outlier_flags.append(bad)
    Xi = X+rng.normal(0,.05,X.shape)
    Ri, ti = R.copy(), t.copy()
    for i in range(2,4):
        Q = exp_so3(rng.normal(0,.018,3))
        Ri[i], ti[i] = Q@R[i], Q@t[i]+rng.normal(0,.04,3)
    # Central finite differences validate the same local update used by the solver.
    i,j,eps = 2,3,1e-6
    A,B = jacobians(Ri[i],ti[i],Xi[j])
    numerical = []
    for k in range(9):
        v = np.zeros(9); v[k] = eps
        values = []
        for sign in [1,-1]:
            Q = exp_so3(sign*v[:3])
            q = Q@Ri[i]@(Xi[j]+sign*v[6:])+Q@ti[i]+sign*v[3:6]
            values.append(project(q))
        numerical.append((values[0]-values[1])/(2*eps))
    jac_error = float(np.max(np.abs(np.column_stack([A,B])-np.array(numerical).T)))
    _, J, r = linearize(Ri, ti, Xi, obs, True)
    step,H,g = schur_step(J,r,12,1e-3)
    schur_error = float(np.max(np.abs(step-np.linalg.solve(H,-g))))
    # Full similarity gauge leaves every pixel unchanged before anchors are applied.
    scale, Q, offset = 2.3, exp_so3(np.array([.2,-.1,.05])), np.array([.2,.1,-.3])
    Xg = scale*(X@Q.T)+offset
    Rg = np.array([rr@Q.T for rr in R])
    tg = scale*t-np.einsum("nij,j->ni", Rg, offset)
    gauge_error = float(np.max(np.abs(residuals(Rg,tg,Xg,heldout)-residuals(R,t,X,heldout))))
    metrics, arrays = {}, dict(truth=X, initial=Xi, train_outliers=np.array(outlier_flags))
    for label, robust in [("least_squares",False),("huber",True)]:
        rr,tt,xx,history = optimize(Ri,ti,Xi,obs,heldout,robust)
        cc = -np.einsum("nji,nj->ni",rr,tt)
        clean = [o for o,bad in zip(obs,outlier_flags) if not bad]
        angles = [np.degrees(np.arccos(np.clip((np.trace(rr[k]@R[k].T)-1)/2,-1,1))) for k in [2,3]]
        metrics[label] = dict(train_all_coordinate_rmse_px=float(np.sqrt(np.mean(residuals(rr,tt,xx,obs)**2))),
            train_clean_coordinate_rmse_px=float(np.sqrt(np.mean(residuals(rr,tt,xx,clean)**2))),
            heldout_clean_coordinate_rmse_px=float(np.sqrt(np.mean(residuals(rr,tt,xx,heldout)**2))),
            median_point_error_m=float(np.median(np.linalg.norm(xx-X,axis=1))),
            mean_free_camera_center_error_m=float(np.mean(np.linalg.norm(cc[2:]-C[2:],axis=1))),
            mean_free_rotation_error_deg=float(np.mean(angles)), iterations=len(history)-1)
        arrays[label+"_points"], arrays[label+"_history"] = xx,history
    # Same observed position, two hidden velocities: deterministic observation is non-Markov.
    dt, velocities = .1, np.array([-.2,.2])
    previous, current, next_pos = -velocities*dt, np.zeros(2), velocities*dt
    reconstructed_v = (current-previous)/dt
    result = dict(seed=23, scope="anchored synthetic BA, first two camera poses fixed",
        cameras=4, free_camera_pose_parameters=12, points=len(X), point_parameters=3*len(X),
        training_observations=len(obs), heldout_observations=len(heldout),
        injected_outliers=int(sum(outlier_flags)), huber_delta_px=1.,
        initial_heldout_coordinate_rmse_px=float(np.sqrt(np.mean(residuals(Ri,ti,Xi,heldout)**2))),
        analytic_jacobian_max_abs_error=jac_error, schur_dense_step_max_abs_difference=schur_error,
        similarity_gauge_max_pixel_difference=gauge_error, optimizers=metrics,
        partial_observation=dict(dt_s=dt,current_position_m=current.tolist(),
            previous_position_m=previous.tolist(),hidden_velocity_m_s=velocities.tolist(),
            next_position_m=next_pos.tolist(),history_recovered_velocity_m_s=reconstructed_v.tolist(),
            position_only_mean_predictor_abs_error_m=float(np.mean(np.abs(next_pos))),
            velocity_noise_std_at_position_sigma_1mm_m_s=float(np.sqrt(2)*.001/dt)))
    assert jac_error < 1e-6 and schur_error < 1e-8 and gauge_error < 1e-10
    assert metrics["huber"]["heldout_clean_coordinate_rmse_px"] < metrics["least_squares"]["heldout_clean_coordinate_rmse_px"]
    np.savez(OUT/"arrays.npz", **arrays)
    (OUT/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    run()
