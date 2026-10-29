import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hidden_hand import HiddenHandEnv, make_bot  # noqa: E402
from hidden_hand.checks.harness import _BotAgent, play_match  # noqa: E402
from muzero.agent import MuZeroAgent  # noqa: E402
from muzero.encoding import FEATURE_DIM, NUM_TOKENS, encode_observation  # noqa: E402
from muzero.mcts import MCTSConfig, run_mcts  # noqa: E402
from muzero.model import ModelConfig, MuZeroNet  # noqa: E402
from muzero.selfplay import ReplayBuffer, play_games  # noqa: E402
from muzero.train import TrainConfig, train_step  # noqa: E402

SMALL = ModelConfig(d_model=16, n_heads=2, n_layers=1, latent_dim=32, hidden_dim=32, n_res_blocks=1, head_dim=16)


def test_encoding_shape_and_no_hidden_info():
    env = HiddenHandEnv(seed=1)
    obs = env.reset()
    feats, pad = encode_observation(obs)
    assert feats.shape == (NUM_TOKENS, FEATURE_DIM) and pad.shape == (NUM_TOKENS,)
    # opponent card values are never encoded unless known to the observer
    opp = feats[4:8]
    assert (opp[:, 5] == 0).all() and (opp[:, 6:16] == 0).all()


def test_search_only_picks_legal_actions():
    net = MuZeroNet(SMALL).eval()
    envs = [HiddenHandEnv(seed=s) for s in range(4)]
    obs = [e.observe(e.current_player) for e in envs]
    enc = [encode_observation(o) for o in obs]
    legal = [e.legal_actions() for e in envs]
    visits, values = run_mcts(
        net, np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]), legal,
        MCTSConfig(num_simulations=8), np.random.default_rng(0), add_noise=True,
    )
    assert values.shape == (4, 2)
    for v, l in zip(visits, legal):
        assert v.sum() == 8 and set(np.flatnonzero(v)) <= set(l)


def test_selfplay_train_and_agent_smoke():
    net = MuZeroNet(SMALL)
    rng = np.random.default_rng(0)
    games = play_games(net.eval(), 6, MCTSConfig(num_simulations=4), rng, seed0=0)
    assert all(len(g) >= 1 and g.rewards[-1].any() for g in games)
    buf = ReplayBuffer(100, td_steps=20, discount=1.0)
    buf.add(games)
    cfg = TrainConfig(unroll=3)
    batch = buf.sample_batch(8, cfg.unroll, rng)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    losses = train_step(net.train(), opt, batch, cfg)
    assert all(np.isfinite(v) for v in losses.values())

    agent = MuZeroAgent(net, simulations=4)
    result = play_match(agent, _BotAgent(make_bot("random", 0)), games=4, seed=0)
    assert result.illegal_actions == 0 and result.truncated == 0
