import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hidden_hand import HiddenHandEnv, make_bot  # noqa: E402
from hidden_hand.checks.harness import _BotAgent, play_match  # noqa: E402
from skynet_base.agent import SkyNetAgent  # noqa: E402
from skynet_base.encoding import encode_observation  # noqa: E402
from skynet_base.mcts import MCTSConfig, run_mcts  # noqa: E402
from skynet_base.model import ModelConfig, MuZeroNet  # noqa: E402
from skynet_base.selfplay import Game, ReplayBuffer, final_ranks, play_games  # noqa: E402
from skynet_base.train import TrainConfig, train_step  # noqa: E402

SMALL = ModelConfig(d_model=16, n_heads=2, n_layers=1, ff_dim=32, latent_dim=32, hidden_dim=32, n_res_blocks=1,
                    head_dim=16, cond_dim=4)


def test_predict_all_egos_shapes_and_ego_dependence():
    net = MuZeroNet(SMALL).eval()
    state = torch.rand(3, SMALL.latent_dim)
    policy, value, winner, rank = net.predict_all_egos(state, torch.tensor([0, 1, 0]), 2)
    bins = 2 * SMALL.support_size + 1
    assert policy.shape == (3, net.prediction.policy[-1].out_features)
    assert value.shape == (3, 2, bins)
    assert winner.shape == (3, 2, SMALL.max_players + 1)
    assert rank.shape == (3, 2, SMALL.max_players)
    # the ego embedding changes the prediction for the same latent
    assert not torch.allclose(value[:, 0], value[:, 1])
    # the mover-relative view of the same latent is a permutation of the root-relative one
    _, v_cur1, _, _ = net.predict_all_egos(state[:1], torch.tensor([1]), 2)
    _, v_cur0, _, _ = net.predict_all_egos(state[:1], torch.tensor([0]), 2)
    assert not torch.allclose(v_cur1, v_cur0)


def test_final_ranks():
    assert final_ranks((3, 10)) == (0, 1)
    assert final_ranks((12, 4)) == (1, 0)
    assert final_ranks((7, 7)) == (0, 0)


def _fake_game(winner: int, scores: tuple[int, int], first: int, T: int = 4) -> Game:
    actor = (first + np.arange(T)) % 2
    return Game(
        feats=np.zeros((T, *encode_observation(HiddenHandEnv(seed=0).reset())[0].shape), dtype=np.float16),
        pad=np.zeros((T, encode_observation(HiddenHandEnv(seed=0).reset())[1].shape[0]), dtype=bool),
        actions=np.zeros(T, dtype=np.int64),
        policies=np.full((T, net_actions()), 1.0 / net_actions(), dtype=np.float32),
        actor=actor.astype(np.int64),
        root_values=np.zeros((T, 2), dtype=np.float32),
        rewards=np.zeros((T, 2), dtype=np.float32),
        final_scores=scores,
        winner=winner,
    )


def net_actions() -> int:
    from hidden_hand.env import ACTION_SPACE_SIZE

    return ACTION_SPACE_SIZE


@pytest.mark.parametrize("first", [0, 1])
def test_outcome_targets(first):
    rng = np.random.default_rng(0)
    K = 6  # longer than the game, so the absorbing tail is covered too
    for winner, scores in ((0, (3, 9)), (1, (9, 3)), (-1, (5, 5))):
        buf = ReplayBuffer(10, td_steps=10, discount=1.0)
        game = _fake_game(winner, scores, first)
        buf.add([game])
        batch = buf.sample_batch(16, K, rng)
        for b in range(16):
            # the root mover is seat actor[s]; recover it from cur_off == 0 at k == 0
            assert batch["cur_off"][b, 0] == 0
            for k in range(K + 1):
                assert batch["cur_off"][b, k] == k % 2  # strict alternation from the root
            # in_game is a prefix of ones
            ig = batch["in_game"][b]
            assert (np.diff(ig) <= 0).all()
        for b in range(16):
            for k in range(K + 1):
                # mover at step k (seat), from the start position implied by in_game length
                s = len(game) - int(batch["in_game"][b].sum())
                mover = (first + s + k) % 2
                for j in range(2):
                    ego = (mover + j) % 2
                    want_w = 2 if winner < 0 else (winner - ego) % 2
                    assert batch["winner"][b, k, j] == want_w
                    assert batch["rank"][b, k, j] == game.final_ranks[ego]


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


def test_train_step_with_aux_heads_and_agent_smoke():
    net = MuZeroNet(SMALL)
    rng = np.random.default_rng(0)
    opps = [None, make_bot("greedy", 0), None, make_bot("random", 1)]
    games = play_games(net.eval(), 4, MCTSConfig(num_simulations=4), rng, seed0=0, opponents=opps)
    buf = ReplayBuffer(100, td_steps=10, discount=0.997)
    buf.add(games)
    cfg = TrainConfig(unroll=3)
    batch = buf.sample_batch(8, cfg.unroll, rng)
    opt = torch.optim.AdamW(net.parameters(), 3e-4)
    losses = train_step(net.train(), opt, batch, cfg, iteration=0)
    assert {"loss_winner", "loss_rank"} <= set(losses)
    assert all(np.isfinite(v) for v in losses.values())
    assert losses["loss_winner"] > 0 and losses["loss_rank"] > 0

    agent = SkyNetAgent(net, simulations=4)
    result = play_match(agent, _BotAgent(make_bot("random", 0)), games=4, seed=0)
    assert result.illegal_actions == 0 and result.truncated == 0


def test_aux_weight_ramp():
    cfg = TrainConfig()
    assert cfg.aux_weights(0) == pytest.approx((0.1, 0.05))
    assert cfg.aux_weights(200) == pytest.approx((0.1, 0.05))
    assert cfg.aux_weights(350) == pytest.approx((0.3, 0.15))
    assert cfg.aux_weights(500) == pytest.approx((0.5, 0.25))
    assert cfg.aux_weights(1999) == pytest.approx((0.5, 0.25))
