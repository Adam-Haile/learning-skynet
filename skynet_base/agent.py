"""SkyNet agent in the interface ``hidden_hand.checks`` expects.

    python -m hidden_hand.checks --muzero muzero.agent:load_muzero --skynet skynet_base.agent:load_skynet ...

``load_skynet`` reads the checkpoint named by ``$SKYNET_CKPT`` (default ``runs/skynet/model.pt``).
Evaluation settings follow the SkyNet paper: 200 simulations with Dirichlet noise (alpha 0.3,
fraction 0.25), greedy over the visit counts.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .encoding import encode_batch
from .mcts import MCTSConfig, run_mcts, select_actions
from .model import MuZeroNet
from .train import TrainConfig, load_checkpoint


class SkyNetAgent:
    def __init__(
        self,
        net: MuZeroNet,
        simulations: int = 200,
        name: str = "skynet",
        seed: int = 0,
        add_noise: bool = True,
        discount: float = TrainConfig.discount,
    ) -> None:
        self.net = net.eval()
        self.cfg = MCTSConfig(num_simulations=simulations, discount=discount)
        self.add_noise = add_noise
        self.name = name
        self.rng = np.random.default_rng(seed)

    def act_batch(self, observations: list[dict], legal_actions: list[list[int]]) -> list[int]:
        feats, pad = encode_batch(observations)
        visits, _ = run_mcts(
            self.net, feats, pad, [list(l) for l in legal_actions], self.cfg, self.rng, add_noise=self.add_noise
        )
        return [int(a) for a in select_actions(visits, 0.0, self.rng)]

    def act(self, obs: dict, legal_actions: list[int]) -> int:
        return self.act_batch([obs], [legal_actions])[0]

    @torch.no_grad()
    def latent(self, observations: list[dict]) -> np.ndarray:
        """The representation output h(o), shape ``[N, latent_dim]`` (before ego conditioning)."""
        feats, pad = encode_batch(observations)
        device = next(self.net.parameters()).device
        return self.net.representation(torch.from_numpy(feats).to(device), torch.from_numpy(pad).to(device)).cpu().numpy()


def load_skynet(path: str | None = None, simulations: int = 200, device: str | None = None) -> SkyNetAgent:
    """``device`` defaults to ``$SKYNET_DEVICE``, else CPU."""
    path = path or os.environ.get("SKYNET_CKPT", "runs/skynet/model.pt")
    device = device or os.environ.get("SKYNET_DEVICE", "cpu")
    return SkyNetAgent(load_checkpoint(path).to(device), simulations=simulations)

