"""Day31: OpenVLA-style token boundaries, unit contracts and contact gating.

Pure NumPy teaching adapter; no OpenVLA/RT-2 weights or model inference.
Vocabulary size and all action statistics below are deliberately synthetic.
"""
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VOCAB_SIZE = 32000  # Teaching constant, not read from a real checkpoint.
EDGES = np.linspace(-1., 1., 256)
CENTERS = (EDGES[:-1]+EDGES[1:])/2
MASK = np.array([True]*6+[False])
STATS = {
    "teaching_base_delta": (np.array([-.02]*3+[-.1]*3+[0.]), np.array([.02]*3+[.1]*3+[1.])),
    "teaching_ten_times": (np.array([-.2]*3+[-1.]*3+[0.]), np.array([.2]*3+[1.]*3+[1.])),
}


def encode(normalized):
    return VOCAB_SIZE-np.digitize(np.clip(normalized, -1., 1.), EDGES)


def decode(tokens):
    tokens = np.asarray(tokens)
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Action token IDs must be integers")
    if np.any((tokens < VOCAB_SIZE-256) | (tokens >= VOCAB_SIZE)):
        raise ValueError("Non-action token; reject before continuous decoding")
    return CENTERS[np.clip(VOCAB_SIZE-tokens-1, 0, len(CENTERS)-1)]


def normalize(action, key):
    low, high = STATS[key]
    return np.where(MASK, 2*(action-low)/(high-low)-1, action)


def unnormalize(normalized, key):
    low, high = STATS[key]
    return np.where(MASK, .5*(normalized+1)*(high-low)+low, normalized)


def rmse(predicted, truth):
    return float(np.sqrt(np.mean((predicted-truth)**2)))


def main():
    out = Path(__file__).resolve().parents[2]/"_assets/3d_learning/day31"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    action = np.array([.012, -.008, .004, .04, -.02, .01, 1.])
    key = "teaching_base_delta"
    normalized = normalize(action, key)
    tokens = encode(normalized)
    restored_norm = decode(tokens)
    restored = unnormalize(restored_norm, key)
    wrong = unnormalize(restored_norm, "teaching_ten_times")
    probes = np.r_[np.linspace(-1., 1., 10001), [-2., 2.]]
    reconstructed = decode(encode(probes))
    within = (probes >= -1) & (probes <= 1)
    quantization_max = float(np.max(np.abs(probes[within]-reconstructed[within])))
    assert quantization_max <= 1/255+1e-12
    assert np.max(np.abs(action[:3]-restored[:3])) <= .04/(2*255)+1e-12
    assert np.allclose(wrong[:6], 10*restored[:6])
    rejected = False
    try:
        decode(np.array([1234]))
    except ValueError:
        rejected = True
    assert rejected
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fine = np.linspace(-.025, .025, 1000)
    axes[0].plot(fine, fine, color="gray", ls="--", label="Identity")
    axes[0].plot(fine, decode(encode(fine)), label="255 center values")
    axes[0].set(xlabel="Normalized input", ylabel="Decoded value", title="Actual edge/center convention")
    axes[0].legend()
    index = np.arange(3)
    for offset, label, values in [(-.24, "Requested", action), (0., "Correct key", restored), (.24, "Wrong key", wrong)]:
        axes[1].bar(index+offset, values[:3]*1000, width=.24, label=label)
    axes[1].set(xticks=index, xticklabels=["dx", "dy", "dz"], ylabel="Translation increment [mm]", title="Same tokens, different statistics")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out/"action_codec.png", dpi=170)
    plt.close(fig)

    # Object-versus-finger tangential speed in mm/s in the same local frame.
    # During contact this is slip speed; without contact the gel cannot observe it.
    count = 20000
    truth = rng.uniform(2., 8., count)
    vision = truth+rng.normal(0., 2., count)
    tactile_valid = truth+rng.normal(0., .2, count)
    tactile_invalid = rng.normal(0., .2, count)  # No contact: estimates gel noise only.
    weight_v, weight_t = 1/2.**2, 1/.2**2
    fuse = lambda a, b: (weight_v*a+weight_t*b)/(weight_v+weight_t)
    fused_contact = fuse(vision, tactile_valid)
    fused_no_contact = fuse(vision, tactile_invalid)
    gated_no_contact = vision.copy()
    values = [rmse(vision, truth), rmse(fused_contact, truth), rmse(fused_no_contact, truth), rmse(gated_no_contact, truth)]
    assert values[1] < values[0]
    assert values[2] > values[0]
    assert values[3] == values[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].bar(["Vision", "Valid\ncontact fusion", "Invalid\nblind fusion", "No contact\ngated"], values,
                color=["C0", "C2", "C3", "C0"])
    axes[0].set(ylabel="Relative-speed RMSE [mm/s]", title="Low variance is not proof of validity")
    levels = np.linspace(0., 7., 141)
    raw_close = np.full_like(levels, .8)
    adapted_close = np.where(levels >= 6., 0., raw_close)
    axes[1].plot(levels, raw_close, "--", label="Policy closing request")
    axes[1].step(levels, adapted_close, where="post", label="Contact-aware adapter")
    axes[1].axvline(6., color="black", ls=":", label="Teaching force limit")
    axes[1].set(xlabel="Measured total normal load [N]", ylabel="Closing rate request [arb.]",
                title="Policy intent + independent contact constraint", ylim=(-.05, 1.))
    axes[1].legend(fontsize=8)
    fig.savefig(out/"contact_validity.png", dpi=170)
    plt.close(fig)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    tool_delta = np.array([.01, 0., 0.])
    base_delta = rotation@tool_delta
    frame_error = float(np.linalg.norm(tool_delta-base_delta)*1000)
    report = dict(scope="Synthetic actions and sensor estimates; no VLA inference",
        environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        contract=dict(key=key, translation="base-frame delta m per action", rotation="base-frame small rotation-vector delta rad per action",
            gripper="0=open, 1=closed target, not force", control_period_s=.1, mask=MASK.tolist(), vocab_size=VOCAB_SIZE),
        action=action.tolist(), normalized=normalized.tolist(), tokens=tokens.tolist(), decoded_normalized=restored_norm.tolist(),
        correct_key_action=restored.tolist(), wrong_key_action=wrong.tolist(),
        quantization=dict(edges=len(EDGES), centers=len(CENTERS), normalized_max_error=quantization_max,
            normalized_bound=1/255, translation_bound_mm=.04/(2*255)*1000,
            out_of_range_input_2_error=float(abs(2-reconstructed[-1])), invalid_token_rejected=rejected),
        translation_error_mm=dict(correct=float(np.linalg.norm(restored[:3]-action[:3])*1000),
            wrong=float(np.linalg.norm(wrong[:3]-action[:3])*1000)),
        frame_counterexample_error_mm=frame_error,
        fusion_rmse_mm_per_s=dict(zip(["vision", "valid_contact", "invalid_blind", "invalid_gated"], values)),
        contact_adapter=dict(limit_N=6., measured_N=7., request_before=.8, request_after=0.,
            caveat="Stops extra closing in toy interface; does not actively remove existing force"))
    (out/"results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
