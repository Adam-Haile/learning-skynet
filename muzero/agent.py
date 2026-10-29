"""MuZero agent in the interface ``hidden_hand.checks`` expects.

    python -m hidden_hand.checks --muzero muzero.agent:load_muzero ...

``load_muzero`` reads the checkpoint named by ``$MUZERO_CKPT`` (default ``runs/muzero/model.pt``).
Evaluation settings follow the README: greedy search, no Dirichlet noise, 16 simulations.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .encoding import encode_batch
from .mcts import MCTSConfig, run_mcts, select_actions
from .model import MuZeroNet
from .train import load_checkpoint


class MuZeroAgent:
    def __init__(self, net: MuZeroNet, simulations: int = 16, name: str = "muzero", seed: int = 0) -> None:
        self.net = net.eval()
        self.cfg = MCTSConfig(num_simulations=simulations)
        self.name = name
        self.rng = np.random.default_rng(seed)

    def act_batch(self, observations: list[dict], legal_actions: list[list[int]]) -> list[int]:
        feats, pad = encode_batch(observations)
        visits, _ = run_mcts(self.net, feats, pad, [list(l) for l in legal_actions], self.cfg, self.rng, add_noise=False)
        return [int(a) for a in select_actions(visits, 0.0, self.rng)]

    def act(self, obs: dict, legal_actions: list[int]) -> int:
        return self.act_batch([obs], [legal_actions])[0]

    @torch.no_grad()
    def latent(self, observations: list[dict]) -> np.ndarray:
        """The representation output h(o), shape ``[N, latent_dim]``."""
        feats, pad = encode_batch(observations)
        return self.net.representation(torch.from_numpy(feats), torch.from_numpy(pad)).numpy()


def load_muzero(path: str | None = None, simulations: int = 16) -> MuZeroAgent:
    path = path or os.environ.get("MUZERO_CKPT", "runs/muzero/model.pt")
    return MuZeroAgent(load_checkpoint(path), simulations=simulations)
