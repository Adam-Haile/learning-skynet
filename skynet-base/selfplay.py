"""Self-play data generation and the replay buffer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hidden_hand.env import ACTION_SPACE_SIZE, HiddenHandConfig, HiddenHandEnv

from .encoding import FEATURE_DIM, NUM_TOKENS, encode_observation
from .mcts import MCTSConfig, run_mcts, select_actions


@dataclass
class Game:
    """One finished game. Row ``t`` is the decision at step ``t`` (``T`` decisions in all)."""

    feats: np.ndarray       # [T, NUM_TOKENS, FEATURE_DIM] float16
    pad: np.ndarray         # [T, NUM_TOKENS] bool
    actions: np.ndarray     # [T] int64
    policies: np.ndarray    # [T, A] float32, normalised MCTS visit counts
    actor: np.ndarray       # [T] int64, player who moved at step t
    root_values: np.ndarray  # [T, 2] float32, [mover, other] from the search at step t
    rewards: np.ndarray     # [T, 2] float32, reward after action t as [player 0, player 1], scaled
    final_scores: tuple[int, int] = (0, 0)
    winner: int = -1

    def __len__(self) -> int:
        return len(self.actions)

    def value_targets(self, td_steps: int, discount: float) -> np.ndarray:
        """n-step return targets as ``[mover, other]`` for every step (bootstrapping from root values)."""
        T = len(self)
        # root values in absolute player order
        absolute = np.where(self.actor[:, None] == 0, self.root_values, self.root_values[:, ::-1])
        targets = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            end = min(t + td_steps, T)
            powers = discount ** np.arange(end - t)
            ret = (powers[:, None] * self.rewards[t:end]).sum(axis=0)
            if t + td_steps < T:
                ret = ret + discount**td_steps * absolute[t + td_steps]
            mover = self.actor[t]
            targets[t] = (ret[mover], ret[1 - mover])
        return targets


def play_games(
    net,
    num_games: int,
    mcts_cfg: MCTSConfig,
    rng: np.random.Generator,
    seed0: int,
    env_config: HiddenHandConfig | None = None,
    reward_scale: float = 10.0,
    temperature: float | callable = 1.0,
    add_noise: bool = True,
) -> list[Game]:
    """Play ``num_games`` self-play games in lockstep, one batched search per move.

    ``temperature`` may be a function of the game step (``step -> temperature``).
    """
    envs = [HiddenHandEnv(seed=seed0 + i, config=env_config) for i in range(num_games)]
    obs = [env.observe(env.current_player) for env in envs]
    rec = [dict(feats=[], pad=[], actions=[], policies=[], actor=[], values=[], rewards=[]) for _ in envs]
    done = [False] * num_games
    temp_fn = temperature if callable(temperature) else (lambda _step: temperature)

    while not all(done):
        active = [i for i in range(num_games) if not done[i]]
        enc = [encode_observation(obs[i]) for i in active]
        feats = np.stack([e[0] for e in enc])
        pad = np.stack([e[1] for e in enc])
        legal = [envs[i].legal_actions() for i in active]
        visits, values = run_mcts(net, feats, pad, legal, mcts_cfg, rng, add_noise)
        for k, i in enumerate(active):
            env = envs[i]
            action = int(select_actions(visits[k:k + 1], temp_fn(env.state.step), rng)[0])
            r = rec[i]
            r["feats"].append(feats[k].astype(np.float16))
            r["pad"].append(pad[k])
            r["actions"].append(action)
            r["policies"].append(visits[k] / visits[k].sum())
            r["actor"].append(env.current_player)
            r["values"].append(values[k])
            obs[i], rewards, terminated, _info = env.step(action)
            r["rewards"].append([rewards["player_0"] / reward_scale, rewards["player_1"] / reward_scale])
            done[i] = terminated

    games = []
    for env, r in zip(envs, rec):
        games.append(
            Game(
                feats=np.stack(r["feats"]),
                pad=np.stack(r["pad"]),
                actions=np.array(r["actions"], dtype=np.int64),
                policies=np.stack(r["policies"]).astype(np.float32),
                actor=np.array(r["actor"], dtype=np.int64),
                root_values=np.stack(r["values"]).astype(np.float32),
                rewards=np.array(r["rewards"], dtype=np.float32),
                final_scores=tuple(env.scores),
                winner=env.winner,
            )
        )
    return games


class ReplayBuffer:
    """FIFO buffer of whole games; samples ``(game, start)`` positions uniformly over decisions."""

    def __init__(self, capacity_games: int, td_steps: int, discount: float) -> None:
        self.capacity = capacity_games
        self.td_steps = td_steps
        self.discount = discount
        self.games: list[Game] = []
        self.value_targets: list[np.ndarray] = []

    def add(self, games: list[Game]) -> None:
        for g in games:
            self.games.append(g)
            self.value_targets.append(g.value_targets(self.td_steps, self.discount))
        excess = len(self.games) - self.capacity
        if excess > 0:
            del self.games[:excess], self.value_targets[:excess]

    def __len__(self) -> int:
        return len(self.games)

    def sample_batch(self, batch_size: int, unroll: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Build a batch of ``unroll``-step training targets.

        Targets for step ``k`` describe the state after ``k`` actions. Past the end of the game the
        state is absorbing: value and reward are 0, the policy is not trained (``policy_mask`` 0).
        """
        lengths = np.array([len(g) for g in self.games])
        gi = rng.choice(len(self.games), size=batch_size, p=lengths / lengths.sum())
        start = np.array([rng.integers(lengths[g]) for g in gi])

        K = unroll
        batch = dict(
            feats=np.zeros((batch_size, NUM_TOKENS, FEATURE_DIM), dtype=np.float32),
            pad=np.zeros((batch_size, NUM_TOKENS), dtype=bool),
            actions=np.zeros((batch_size, K), dtype=np.int64),
            policy=np.zeros((batch_size, K + 1, ACTION_SPACE_SIZE), dtype=np.float32),
            policy_mask=np.zeros((batch_size, K + 1), dtype=np.float32),
            value=np.zeros((batch_size, K + 1, 2), dtype=np.float32),
            reward=np.zeros((batch_size, K + 1, 2), dtype=np.float32),
        )
        for b, (g, s) in enumerate(zip(gi, start)):
            game, vt = self.games[g], self.value_targets[g]
            T = len(game)
            batch["feats"][b] = game.feats[s]
            batch["pad"][b] = game.pad[s]
            for k in range(K + 1):
                t = s + k
                if k > 0:
                    # transition t-1 -> t, rewards for [actor of t-1, other]
                    actor = game.actor[t - 1] if t - 1 < T else 0
                    if t - 1 < T:
                        r = game.rewards[t - 1]
                        batch["reward"][b, k] = (r[actor], r[1 - actor])
                if t < T:
                    batch["policy"][b, k] = game.policies[t]
                    batch["policy_mask"][b, k] = 1.0
                    batch["value"][b, k] = vt[t]
                if k < K:
                    batch["actions"][b, k] = game.actions[t] if t < T else 0
        return batch
