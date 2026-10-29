"""MuZero networks: representation h, dynamics g and prediction f.

    s_0           = h(observation)
    s_{k+1}, r_k  = g(s_k, a_k)
    p_k, v_k      = f(s_k)

Two-player convention (the game is *not* zero-sum, so values are never negated). Every value and
reward is a 2-vector ``[mover, other]``:

* ``v_k``: expected return for ``[the player to move at s_k, the other player]``;
* ``r_k``: reward paid by the transition for ``[the player who acted, the other player]``.

Players strictly alternate, so moving one node up the tree swaps the two entries (see ``mcts.py``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from hidden_hand.env import ACTION_SPACE_SIZE

from .encoding import FEATURE_DIM, NUM_TOKENS, NUM_TYPES, TOKEN_SLOTS, TOKEN_TYPES


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    latent_dim: int = 128
    hidden_dim: int = 256
    n_res_blocks: int = 2
    head_dim: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + x.detach() * (1.0 - scale)


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale each latent vector into [0, 1] (the MuZero hidden-state normalisation)."""
    lo = x.min(dim=-1, keepdim=True).values
    hi = x.max(dim=-1, keepdim=True).values
    return (x - lo) / (hi - lo).clamp_min(1e-5)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class Representation(nn.Module):
    """Transformer over the observation tokens, pooled through a CLS token into one latent vector."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.input = nn.Linear(FEATURE_DIM, d)
        self.type_emb = nn.Embedding(NUM_TYPES, d)
        self.slot_emb = nn.Embedding(NUM_TOKENS, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.latent_dim)
        self.register_buffer("types", torch.as_tensor(TOKEN_TYPES), persistent=False)
        self.register_buffer("slots", torch.arange(NUM_TOKENS), persistent=False)
        assert len(TOKEN_SLOTS) == NUM_TOKENS

    def forward(self, feats: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        x = self.input(feats) + self.type_emb(self.types) + self.slot_emb(self.slots)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        return min_max_normalize(self.out(self.norm(x[:, 0])))


class Dynamics(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(cfg.latent_dim + ACTION_SPACE_SIZE, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.hidden_dim) for _ in range(cfg.n_res_blocks)])
        self.next_state = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.reward = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.head_dim), nn.ReLU(), nn.Linear(cfg.head_dim, 2))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = torch.nn.functional.one_hot(action, ACTION_SPACE_SIZE).to(state.dtype)
        h = self.blocks(self.inp(torch.cat([state, onehot], dim=-1)))
        return min_max_normalize(self.next_state(h)), self.reward(h)


class Prediction(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.hidden_dim) for _ in range(cfg.n_res_blocks)])
        self.policy = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.head_dim), nn.ReLU(), nn.Linear(cfg.head_dim, ACTION_SPACE_SIZE))
        self.value = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.head_dim), nn.ReLU(), nn.Linear(cfg.head_dim, 2))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.trunk(state))
        return self.policy(h), self.value(h)


class MuZeroNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.representation = Representation(self.cfg)
        self.dynamics = Dynamics(self.cfg)
        self.prediction = Prediction(self.cfg)

    def initial_inference(self, feats, pad):
        """Observation -> ``(state, policy_logits, value)``."""
        state = self.representation(feats, pad)
        logits, value = self.prediction(state)
        return state, logits, value

    def recurrent_inference(self, state, action):
        """``(state, action)`` -> ``(next_state, reward, policy_logits, value)``."""
        next_state, reward = self.dynamics(state, action)
        logits, value = self.prediction(next_state)
        return next_state, reward, logits, value
