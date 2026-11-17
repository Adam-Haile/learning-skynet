"""SkyNet networks: representation h, dynamics g and ego-conditioned prediction f.

    s_0                  = h(observation)
    s_{k+1}, r_k         = g(s_k, a_k)
    z_k^e                = [s_k, e_ego(e), e_current(cur_k - e), e_n(n)]     (ego conditioning)
    p_k, v_k^e, w_k^e, q_k^e = f(z_k^e)

Seats are offsets from the *root observer* (the player whose observation built ``s_0``):
``cur_k`` is the seat to move at ``s_k`` and ``e`` the ego, the seat whose outcome ``f`` estimates.
Dynamics never sees the ego, so one latent trajectory serves every ego. ``f`` returns

* ``v``: the ego's expected return (categorical support logits);
* ``w``: winner logits, class ``j`` = "the seat ``j`` places after the ego wins", last class = draw;
* ``q``: rank logits for the ego's final rank (0 = best, tied seats share the best rank);
* ``p``: policy logits (only the ``ego == mover`` row is used).

The game is *not* zero-sum, so values are never negated. ``predict_all_egos`` evaluates every ego in
mover-relative order, so for two players values and rewards are 2-vectors ``[mover, other]``:

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
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024      # transformer feed-forward hidden size
    latent_dim: int = 512
    hidden_dim: int = 256
    n_res_blocks: int = 2
    head_dim: int = 64
    support_size: int = 10  # categorical value/reward support is [-support_size, support_size]
    max_players: int = 2    # sizes the seat embeddings and the winner / rank heads
    cond_dim: int = 16      # size of each ego-conditioning embedding (three are appended)

    def to_dict(self) -> dict:
        return asdict(self)


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + x.detach() * (1.0 - scale)


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale each latent vector into [0, 1] (the MuZero hidden-state normalisation)."""
    lo = x.min(dim=-1, keepdim=True).values
    hi = x.max(dim=-1, keepdim=True).values
    return (x - lo) / (hi - lo).clamp_min(1e-5)


TRANSFORM_EPS = 1e-3


def scalar_transform(x: torch.Tensor) -> torch.Tensor:
    """``h(x)``: the invertible value-compressing transform from the MuZero paper."""
    return torch.sign(x) * (torch.sqrt(x.abs() + 1.0) - 1.0) + TRANSFORM_EPS * x


def inverse_scalar_transform(y: torch.Tensor) -> torch.Tensor:
    """``h^-1(y)``."""
    eps = TRANSFORM_EPS
    return torch.sign(y) * (((torch.sqrt(1.0 + 4.0 * eps * (y.abs() + 1.0 + eps)) - 1.0) / (2.0 * eps)) ** 2 - 1.0)


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Two-hot target ``[..., 2 * support_size + 1]`` for scalars ``x`` (``h`` is applied first)."""
    y = scalar_transform(x).clamp(-support_size, support_size)
    lo = y.floor()
    p_hi = y - lo
    lo_idx = (lo + support_size).long()
    hi_idx = (lo_idx + 1).clamp_max(2 * support_size)
    target = torch.zeros(*x.shape, 2 * support_size + 1, dtype=x.dtype, device=x.device)
    target.scatter_add_(-1, lo_idx.unsqueeze(-1), (1.0 - p_hi).unsqueeze(-1))
    target.scatter_add_(-1, hi_idx.unsqueeze(-1), p_hi.unsqueeze(-1))
    return target


def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Expected scalar of categorical ``logits [..., 2 * support_size + 1]`` (``h^-1`` of the mean)."""
    support = torch.arange(-support_size, support_size + 1, dtype=logits.dtype, device=logits.device)
    return inverse_scalar_transform((torch.softmax(logits, dim=-1) * support).sum(-1))


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
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim, dropout=0.0, batch_first=True, norm_first=True
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
        self.bins = 2 * cfg.support_size + 1
        self.reward = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.head_dim), nn.ReLU(), nn.Linear(cfg.head_dim, 2 * self.bins))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(next_state, reward_logits [B, 2, bins])``."""
        onehot = torch.nn.functional.one_hot(action, ACTION_SPACE_SIZE).to(state.dtype)
        h = self.blocks(self.inp(torch.cat([state, onehot], dim=-1)))
        return min_max_normalize(self.next_state(h)), self.reward(h).view(-1, 2, self.bins)


class EgoConditioning(nn.Module):
    """Append ``[e_ego, e_current, e_n]`` to a latent state.

    * ``e_ego``: the ego's seat offset from the root observer;
    * ``e_current``: the seat to move, relative to the ego (0 = it is the ego's turn);
    * ``e_n``: the number of players in the game.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ego_emb = nn.Embedding(cfg.max_players, cfg.cond_dim)
        self.cur_emb = nn.Embedding(cfg.max_players, cfg.cond_dim)
        self.n_emb = nn.Embedding(cfg.max_players + 1, cfg.cond_dim)

    def forward(self, state: torch.Tensor, ego_off: torch.Tensor, cur_off: torch.Tensor, n_players: int) -> torch.Tensor:
        n = torch.full_like(ego_off, n_players)
        return torch.cat(
            [state, self.ego_emb(ego_off), self.cur_emb((cur_off - ego_off) % n_players), self.n_emb(n)], dim=-1
        )


class Prediction(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.condition = EgoConditioning(cfg)
        in_dim = cfg.latent_dim + 3 * cfg.cond_dim
        self.trunk = nn.Sequential(nn.Linear(in_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.hidden_dim) for _ in range(cfg.n_res_blocks)])
        self.bins = 2 * cfg.support_size + 1

        def head(out: int) -> nn.Module:
            return nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.head_dim), nn.ReLU(), nn.Linear(cfg.head_dim, out))

        self.policy = head(ACTION_SPACE_SIZE)
        self.value = head(self.bins)
        self.winner = head(cfg.max_players + 1)
        self.rank = head(cfg.max_players)

    def forward(self, state, ego_off, cur_off, n_players: int):
        """Returns ``(policy [B, A], value [B, bins], winner [B, max_players + 1], rank [B, max_players])`` logits."""
        h = self.blocks(self.trunk(self.condition(state, ego_off, cur_off, n_players)))
        return self.policy(h), self.value(h), self.winner(h), self.rank(h)


class MuZeroNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.representation = Representation(self.cfg)
        self.dynamics = Dynamics(self.cfg)
        self.prediction = Prediction(self.cfg)

    def to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        return support_to_scalar(logits, self.cfg.support_size)

    def predict_all_egos(self, state: torch.Tensor, cur_off: torch.Tensor | None = None, n_players: int = 2):
        """Run ``f`` once per ego, egos in mover-relative order (ego ``j`` is seat ``cur_off + j``).

        Returns raw logits ``(policy [B, A], value [B, n, bins], winner [B, n, max_players + 1],
        rank [B, n, max_players])``; ``policy`` comes from the ego == mover row.
        """
        assert 1 <= n_players <= self.cfg.max_players
        B, n = state.shape[0], n_players
        if cur_off is None:
            cur_off = torch.zeros(B, dtype=torch.long, device=state.device)
        cur_off = torch.as_tensor(cur_off, dtype=torch.long, device=state.device) % n
        j = torch.arange(n, device=state.device)
        ego_off = (cur_off[:, None] + j[None, :]) % n                       # [B, n]
        flat = state.repeat_interleave(n, dim=0)                            # [B * n, latent]
        policy, value, winner, rank = self.prediction(
            flat, ego_off.reshape(-1), cur_off.repeat_interleave(n), n_players
        )
        return (
            policy.view(B, n, -1)[:, 0],
            value.view(B, n, -1),
            winner.view(B, n, -1),
            rank.view(B, n, -1),
        )

    def initial_inference(self, feats, pad, n_players: int = 2, raw: bool = False):
        """Observation -> ``(state, policy_logits, value)``; ``value`` is ``[B, n]`` as
        ``[mover, other, ...]`` (support logits if ``raw``)."""
        state = self.representation(feats, pad)
        logits, value, _w, _r = self.predict_all_egos(state, None, n_players)
        return state, logits, (value if raw else self.to_scalar(value))

    def recurrent_inference(self, state, action, cur_off=None, n_players: int = 2, raw: bool = False):
        """``(state, action)`` -> ``(next_state, reward, policy_logits, value)``; scalars unless ``raw``.

        ``cur_off`` is the seat to move at ``next_state`` relative to the root observer (``[B]``).
        """
        next_state, reward = self.dynamics(state, action)
        logits, value, _w, _r = self.predict_all_egos(next_state, cur_off, n_players)
        if raw:
            return next_state, reward, logits, value
        return next_state, self.to_scalar(reward), logits, self.to_scalar(value)
