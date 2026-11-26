"""Day21: auditable grasp filters, latency, and a tiny contrastive calculation.

Planar TCP-disc sweeps and position-only 2R IK; not full-hand/arm collision,
force closure, FoundationPose inference, or a physical grasp success test.
"""
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def segment_distance(point, a, b):
    edge = b-a
    u = np.clip(np.dot(point-a, edge)/np.dot(edge, edge), 0., 1.)
    return float(np.linalg.norm(point-(a+u*edge)))


def ik_2r(p, lengths=(0.45, 0.4)):
    l1, l2 = lengths
    cosine = (p @ p-l1*l1-l2*l2)/(2*l1*l2)
    if not -1 <= cosine <= 1:
        return []
    results = []
    for q2 in (np.arccos(cosine), -np.arccos(cosine)):
        q1 = np.arctan2(p[1], p[0])-np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
        results.append([float(q1), float(q2)])
    return results


def evaluate(candidate, sigma):
    center = np.array(candidate["center"])
    pre = center+np.array(candidate["pre_offset"])
    obstacle = np.array([0.38, 0.2])
    margin = 3*sigma
    required_width = candidate["span"]+0.01+2*margin
    clearance = segment_distance(obstacle, pre, center)-0.035-0.015-margin
    friction_limit = np.degrees(np.arctan(0.5))
    branches = ik_2r(center)
    checks = dict(width=0.005 <= required_width <= 0.1,
        antipodal=candidate["normal_angle_deg"] <= friction_limit,
        position_ik=len(branches) > 0,
        tcp_approach=clearance > 0,
        freshness=candidate["age_s"] <= 0.1,
        contact_center=np.linalg.norm(center-[0.48, 0.2]) < 0.01)
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(name=candidate["name"], score=candidate["score"], required_width_m=required_width,
        approach_clearance_m=clearance, ik_branches=len(branches), checks=checks,
        feasible=all(checks.values()))


def info_nce(positive_angle_deg, tau):
    angles = np.radians([positive_angle_deg, 90, 180])
    logits = np.cos(angles)/tau
    shifted = logits-logits.max()
    prob = np.exp(shifted)/np.exp(shifted).sum()
    return float(-np.log(prob[0])), prob


def main():
    out = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day21"
    out.mkdir(parents=True, exist_ok=True)
    common = dict(center=[.48, .2], span=.06, pre_offset=[.15, 0.], normal_angle_deg=0., age_s=.02)
    candidates = []
    for name, score, changes in [
        ("wide", .99, dict(span=.12)),
        ("left_approach", .98, dict(pre_offset=[-.15, 0.])),
        ("right_approach", .93, {}),
        ("bad_normals", .97, dict(normal_angle_deg=40.)),
        ("far_center", .96, dict(center=[1.2, .2])),
        ("stale", .95, dict(age_s=.28)),
    ]:
        candidates.append(dict(common, name=name, score=score) | changes)
    baseline = [evaluate(c, .001) for c in candidates]
    uncertain = [evaluate(c, .008) for c in candidates]
    selected = max((x for x in baseline if x["feasible"]), key=lambda x:x["score"])
    assert selected["name"] == "right_approach"
    assert not any(x["feasible"] for x in uncertain)
    # Same geometry, one independent contrast: only candidate score changes.
    reversed_scores = [{**c, "score": 1-c["score"]} for c in candidates]
    score_control = [evaluate(c, .001) for c in reversed_scores]
    assert [x["feasible"] for x in baseline] == [x["feasible"] for x in score_control]

    contrast = []
    for tau in (1., .1):
        loss, prob = info_nce(5., tau)
        contrast.append(dict(tau=tau, loss=loss, positive_probability=float(prob[0])))
    bad_loss, _ = info_nce(180., .1)
    delays = np.array([.02, .12, .28])
    stale_error_mm = 1000*.2*delays
    results = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__, device="CPU"),
        baseline_sigma_m=.001, candidates=baseline, selected=selected["name"],
        high_uncertainty_sigma_m=.008, high_uncertainty_feasible=[x["name"] for x in uncertain if x["feasible"]],
        high_uncertainty_required_width_m=uncertain[2]["required_width_m"],
        score_flip_feasibility_unchanged=True,
        latency=dict(velocity_m_s=.2, age_s=delays.tolist(), stale_error_mm=stale_error_mm.tolist()),
        contrastive=contrast, inappropriate_positive_loss=bad_loss,
        boundary="No trained model; independent synthetic filter demo. No full arm/hand collision, contact dynamics, actual detection or pose inference.")
    (out/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(11.3, 4.5), layout="constrained")
    ax[0].add_patch(plt.Rectangle((.42,.17), .12,.06, color="cornflowerblue", alpha=.5, label="object footprint"))
    ax[0].add_patch(plt.Circle((.38,.2), .035, color="gray", alpha=.6, label="obstacle"))
    ax[0].add_patch(plt.Circle((.38,.2), .035+.015+.003, fill=False, linestyle="--", label="TCP + uncertainty inflation"))
    for c, color in ((candidates[1], "red"), (candidates[2], "green")):
        center = np.array(c["center"]); pre = center+np.array(c["pre_offset"])
        ax[0].plot([pre[0], center[0]], [pre[1], center[1]], color=color, marker="o", label=c["name"])
    ax[0].set(xlim=(.25,.72), ylim=(.05,.4), xlabel="base x (m)", ylabel="base y (m)", aspect="equal", title="Same object pose; different approaches")
    ax[0].legend(fontsize=7)
    keys = list(baseline[0]["checks"])
    grid = np.array([[int(x["checks"][key]) for key in keys] for x in baseline])
    ax[1].imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax[1].set_xticks(range(len(keys)), keys, rotation=40, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(candidates)), [c["name"] for c in candidates], fontsize=8)
    for i in range(len(candidates)):
        for j in range(len(keys)):
            ax[1].text(j,i,"pass" if grid[i,j] else "fail",ha="center",va="center",fontsize=8)
    ax[1].set_title("Hard feasibility gates before ranking")
    fig.savefig(out/"grasp-filters.png",dpi=170); plt.close(fig)
    fig, ax = plt.subplots(1,3,figsize=(12.2,3.9),layout="constrained")
    sigmas = np.linspace(0,.01,101)
    ax[0].plot(sigmas*1000, (.06+.01+6*sigmas)*1000)
    ax[0].axhline(100,color="red",linestyle="--",label="maximum opening")
    ax[0].set(xlabel="assumed position sigma (mm)",ylabel="required opening (mm)",title="Uncertainty changes feasibility")
    ax[0].legend(fontsize=8)
    ax[1].bar(delays*1000, stale_error_mm,width=35)
    ax[1].set(xlabel="pose age (ms)",ylabel="stale target error (mm)",title="Motion turns delay into error")
    angles=np.linspace(0,180,181)
    for tau in (1.,.1):
        ax[2].plot(angles,[info_nce(a,tau)[0] for a in angles],label=f"temperature {tau}")
    ax[2].set(xlabel="positive-pair angle (deg)",ylabel="InfoNCE loss",title="Pair semantics come before temperature")
    ax[2].legend(fontsize=8)
    fig.savefig(out/"uncertainty-latency-learning.png",dpi=170); plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
