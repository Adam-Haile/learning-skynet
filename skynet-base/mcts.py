"""Batched Monte-Carlo tree search for MuZero (one tree per game, one network call per simulation).

Search treats the game as a two-player, general-sum tree. Node values are 2-vectors
``[player to move at the node, the other player]`` and the reward on the edge into a node is
``[player who acted, the other player]``. Players alternate, so going from a child to its parent
*swaps* the two entries; it never negates them:

    Q(parent, a)  = r_edge[0] + gamma * mean_value(child)[1]
    G_parent      = r_edge    + gamma * swap(G_child)

Chance (the next draw) is not modelled explicitly; as in plain MuZero the dynamics model has to
absorb it into the latent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from hidden_hand.env import ACTION_SPACE_SIZE


@dataclass
class MCTSConfig:
    num_simulations: int = 16
    c1: float = 1.25
    c2: float = 19652.0
    discount: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25


class _MinMax:
    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf

    def update(self, x: float) -> None:
        self.lo = min(self.lo, x)
        self.hi = max(self.hi, x)

    def normalize(self, x: float) -> float:
        return (x - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.0


class _Node:
    __slots__ = ("priors", "allowed", "children", "visits", "v0", "v1", "r0", "r1", "state")

    def __init__(self) -> None:
        self.priors = None
        self.allowed = None
        self.children = None
        self.visits = 0
        self.v0 = 0.0  # summed value, player to move at this node
        self.v1 = 0.0  # summed value, the other player
        self.r0 = 0.0  # reward on the edge into this node, for the player who acted
        self.r1 = 0.0
        self.state = None

    @property
    def expanded(self) -> bool:
        return self.priors is not None

    def expand(self, priors: np.ndarray, allowed: list[int]) -> None:
        self.priors = priors
        self.allowed = allowed
        self.children = [None] * ACTION_SPACE_SIZE


def _select(node: _Node, cfg: MCTSConfig, mm: _MinMax) -> int:
    pb_c = cfg.c1 + math.log((node.visits + cfg.c2 + 1.0) / cfg.c2)
    sqrt_n = math.sqrt(node.visits)
    best, best_score = -1, -math.inf
    for a in node.allowed:
        child = node.children[a]
        if child is None or child.visits == 0:
            q, n = 0.0, 0
        else:
            n = child.visits
            q = mm.normalize(child.r0 + cfg.discount * child.v1 / n)
        score = q + node.priors[a] * sqrt_n / (1 + n) * pb_c
        if score > best_score:
            best, best_score = a, score
    return best


def _softmax(logits: np.ndarray, allowed: list[int] | None = None) -> np.ndarray:
    if allowed is not None:
        masked = np.full_like(logits, -np.inf)
        masked[allowed] = logits[allowed]
        logits = masked
    e = np.exp(logits - logits.max())
    return e / e.sum()


@torch.no_grad()
def run_mcts(
    net,
    feats: np.ndarray,
    pad: np.ndarray,
    legal: list[list[int]],
    cfg: MCTSConfig,
    rng: np.random.Generator,
    add_noise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Search from each of ``B`` root observations.

    Returns ``(visit_counts [B, A], root_values [B, 2])``; the root value is for
    ``[player to move, other player]``.
    """
    B = len(legal)
    state, logits, value = net.initial_inference(torch.from_numpy(feats), torch.from_numpy(pad))
    logits, value = logits.numpy(), value.numpy()

    roots, minmax = [], []
    for b in range(B):
        root = _Node()
        root.state = state[b]
        priors = _softmax(logits[b], legal[b])
        if add_noise:
            noise = np.zeros(ACTION_SPACE_SIZE)
            noise[legal[b]] = rng.dirichlet([cfg.dirichlet_alpha] * len(legal[b]))
            priors = (1 - cfg.dirichlet_fraction) * priors + cfg.dirichlet_fraction * noise
        root.expand(priors, list(legal[b]))
        root.visits, root.v0, root.v1 = 1, float(value[b, 0]), float(value[b, 1])
        roots.append(root)
        mm = _MinMax()
        mm.update(float(value[b, 0]))
        minmax.append(mm)

    all_actions = list(range(ACTION_SPACE_SIZE))
    for _ in range(cfg.num_simulations):
        paths, parents, actions = [], [], []
        for b in range(B):
            node, path = roots[b], [roots[b]]
            while True:
                a = _select(node, cfg, minmax[b])
                child = node.children[a]
                if child is None:
                    child = node.children[a] = _Node()
                path.append(child)
                if not child.expanded:
                    parents.append(node)
                    actions.append(a)
                    break
                node = child
            paths.append(path)

        states = torch.stack([p.state for p in parents])
        next_state, reward, logits, value = net.recurrent_inference(states, torch.tensor(actions))
        reward, logits, value = reward.numpy(), logits.numpy(), value.numpy()
        policy = np.exp(logits - logits.max(axis=1, keepdims=True))
        policy /= policy.sum(axis=1, keepdims=True)

        for b in range(B):
            path = paths[b]
            leaf = path[-1]
            leaf.state = next_state[b]
            leaf.r0, leaf.r1 = float(reward[b, 0]), float(reward[b, 1])
            leaf.expand(policy[b], all_actions)
            g0, g1 = float(value[b, 0]), float(value[b, 1])
            for node in reversed(path):
                node.v0 += g0
                node.v1 += g1
                node.visits += 1
                if node is path[0]:
                    break
                # step to the parent's point of view: its own reward plus the swapped child return
                g0, g1 = node.r0 + cfg.discount * g1, node.r1 + cfg.discount * g0
                minmax[b].update(g0)

    visits = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.float32)
    root_values = np.zeros((B, 2), dtype=np.float32)
    for b, root in enumerate(roots):
        for a, child in enumerate(root.children):
            if child is not None:
                visits[b, a] = child.visits
        root_values[b] = (root.v0 / root.visits, root.v1 / root.visits)
    return visits, root_values


def select_actions(visits: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from visit counts ``** (1 / temperature)``; ``temperature == 0`` is greedy."""
    out = np.zeros(len(visits), dtype=np.int64)
    for b, v in enumerate(visits):
        if temperature <= 1e-6:
            best = np.flatnonzero(v == v.max())
            out[b] = rng.choice(best)
        else:
            p = v ** (1.0 / temperature)
            out[b] = rng.choice(len(v), p=p / p.sum())
    return out
