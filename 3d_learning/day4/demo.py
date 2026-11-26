"""Day 4: BC coverage, DAgger, tabular Q-learning and system identification.

Python 3.10+, standard library, CPU, no downloads or robot connection.
Run: python3 demo.py --seed 7 --episodes 1200
This is a teaching example, not a fair algorithm benchmark.
"""
import argparse
import random

ACTIONS = (-1, 0, 1)


class LineEnv:
    """x in {0,...,6}, action in {-1,0,1}; goal=6; one-step gain=2."""

    def __init__(self, disturb=False, max_steps=8):
        self.disturb = disturb
        self.max_steps = max_steps
        self.reset()

    def reset(self, start=0):
        if start not in range(6):
            raise ValueError("start must be a nonterminal integer in [0,5]")
        self.x, self.t = start, 0
        return {"position": self.x}

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError("action must be -1, 0 or 1")
        if self.x == 6 or self.t >= self.max_steps:
            raise RuntimeError("episode ended; call reset")
        wind = 1 if self.disturb and self.t == 0 else 0
        self.x = max(0, min(6, self.x + 2 * action + wind))
        self.t += 1
        terminated = self.x == 6
        truncated = self.t >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.02
        return {"position": self.x}, reward, terminated, truncated, {}


def rollout(policy, disturb=False, start=0):
    env = LineEnv(disturb=disturb)
    obs = env.reset(start)
    path, visited, rewards = [obs["position"]], [], []
    while True:
        x = obs["position"]
        visited.append(x)
        obs, r, terminal, timeout, _ = env.step(policy(x))
        path.append(obs["position"])
        rewards.append(r)
        if terminal or timeout:
            return terminal, path, visited, rewards


def bc_policy(labels):
    # A deliberately weak memorizing classifier: unseen observations -> stop.
    return lambda x: labels.get(x, 0)


def fit_dagger(initial_labels, rounds=2):
    labels = dict(initial_labels)
    for _ in range(rounds):
        _, _, visited, _ = rollout(bc_policy(labels), disturb=True)
        # Expert is queried on the learner's visited states; aggregate and refit.
        # Here the expert always moves right. No reward is used for this update.
        for x in visited:
            labels[x] = 1
    return labels


def fit_q(seed=7, episodes=1200):
    rng = random.Random(seed)
    q = [[0.0 for _ in ACTIONS] for _ in range(7)]
    interactions = 0
    for _ in range(episodes):
        env = LineEnv()
        obs = env.reset(rng.randrange(6))
        while True:
            x = obs["position"]
            i = rng.randrange(3) if rng.random() < 0.25 else max(
                range(3), key=lambda j: q[x][j]
            )
            obs, r, terminal, timeout, _ = env.step(ACTIONS[i])
            # Time-limit truncation is treated as an external sampling cutoff.
            # Only true goal termination removes the bootstrap term.
            target = r if terminal else r + 0.9 * max(q[obs["position"]])
            q[x][i] += 0.3 * (target - q[x][i])
            interactions += 1
            if terminal or timeout:
                break
    policy = lambda x: ACTIONS[max(range(3), key=lambda j: q[x][j])]
    return policy, q, interactions


def fit_gain(samples):
    # Unclipped transitions x_next = x + k*a; least squares without intercept.
    den = sum(a * a for x, a, xn in samples)
    if den == 0:
        raise ValueError("nonzero actions are required to identify the gain")
    return sum(a * (xn - x) for x, a, xn in samples) / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1200)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    labels = {0: 1, 2: 1, 4: 1}
    bc = bc_policy(labels)
    offline_error = sum(bc(x) != a for x, a in labels.items()) / len(labels)
    nominal = rollout(bc)
    disturbed = rollout(bc, disturb=True)
    dagger_labels = fit_dagger(labels)
    dagger = rollout(bc_policy(dagger_labels), disturb=True)
    q_policy, q, interactions = fit_q(args.seed, args.episodes)
    q_result = rollout(q_policy, disturb=True)

    print(f"BC demonstration action error: {offline_error:.1f}")
    for name, result in [("BC nominal", nominal), ("BC disturbed", disturbed),
                         ("DAgger disturbed", dagger), ("Q-learning disturbed", q_result)]:
        success, path, _, rewards = result
        ret = sum(0.9 ** t * r for t, r in enumerate(rewards))
        print(f"{name:22s}: success={success}, path={path}, return={ret:.6f}")
    print(f"DAgger labels: {sorted(dagger_labels.items())}")
    print(f"Q training interactions: {interactions}; greedy actions at x=0..5: "
          f"{[q_policy(x) for x in range(6)]}")
    print(f"Q(0,+1)={q[0][2]:.6f}; analytic optimal return=0.772000")

    k = fit_gain([(0, 1, 2), (3, -1, 1), (1, 1, 3)])
    print(f"Identified gain on unclipped samples: {k:.1f}")
    for horizon in (1, 5, 10):
        true_x, wrong_x, fitted_x = 2.0 * horizon, 1.5 * horizon, k * horizon
        print(f"rollout H={horizon:2d}: true={true_x:g}, wrong={wrong_x:g}, "
              f"error={abs(true_x-wrong_x):g}, fitted_error={abs(true_x-fitted_x):g}")

    assert offline_error == 0 and nominal[0] and not disturbed[0]
    assert dagger[0] and dagger[1] == [0, 3, 5, 6]
    assert abs(k - 2.0) < 1e-12
    assert abs(sum(0.9 ** t * r for t, r in enumerate(nominal[3])) - 0.772) < 1e-12
    print("Deterministic teaching checks: PASS (Q-learning result is measured, not asserted).")


if __name__ == "__main__":
    main()
