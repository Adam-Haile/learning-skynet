"""Rule and interface tests for Hidden Hand."""
from __future__ import annotations

import random

import pytest

from hidden_hand import (
    ACTION_SPACE_SIZE,
    CALL_ACTION,
    KEEP_ACTION,
    PEEK_ACTION,
    TRADE_BASE,
    UNKNOWN_VALUE,
    HiddenHandConfig,
    HiddenHandEnv,
    make_bot,
)


def play_random(seed: int, config: HiddenHandConfig | None = None) -> tuple[HiddenHandEnv, int]:
    env = HiddenHandEnv(seed=seed, config=config)
    rng = random.Random(seed)
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(rng.choice(env.legal_actions()))
        steps += 1
    return env, steps


def test_deterministic_under_seed():
    a, _ = play_random(7)
    b, _ = play_random(7)
    assert a.get_full_state() == b.get_full_state()
    c, _ = play_random(8)
    assert a.get_full_state() != c.get_full_state()


def test_episodes_terminate_within_limit():
    cfg = HiddenHandConfig()
    for seed in range(300):
        env, steps = play_random(seed)
        assert env.game_over
        assert steps <= 2 * cfg.max_turns_per_player
        assert env.legal_actions() == []


def test_illegal_action_raises():
    env = HiddenHandEnv(seed=0)
    with pytest.raises(ValueError):
        env.step(CALL_ACTION)  # too early to call
    me = env.current_player
    for pos in range(4):
        env.state.hands[me].known_by[pos][me] = True
    with pytest.raises(ValueError):
        env.step(PEEK_ACTION)  # nothing left to peek
    with pytest.raises(ValueError):
        env.step(ACTION_SPACE_SIZE)


def test_observation_never_leaks_hidden_cards():
    for seed in range(50):
        env = HiddenHandEnv(seed=seed)
        rng = random.Random(seed)
        while not env.game_over:
            for observer in (0, 1):
                obs = env.observe(observer)
                for owner, view in obs["board_views"].items():
                    for pos, (value, visible) in enumerate(zip(view["values"], view["visible_mask"])):
                        knows = env.state.hands[owner].known_by[pos][observer]
                        assert bool(visible) == knows
                        assert value == (env.state.hands[owner].cards[pos] if knows else UNKNOWN_VALUE)
                assert "hidden" not in obs
                drawn = obs["tokens"]["decision_token"][1]
                if observer == env.current_player:
                    assert drawn == env.state.drawn
                else:
                    assert drawn == UNKNOWN_VALUE and obs["drawn_card"] is None
            env.step(rng.choice(env.legal_actions()))


def test_token_schema_matches_repo():
    obs = HiddenHandEnv(seed=3).observe(0)
    tokens = obs["tokens"]
    assert len(tokens["board_tokens"]) == 8 and all(len(t) == 4 for t in tokens["board_tokens"])
    assert tokens["board_tokens"][0][0] == 0  # observer's board comes first
    assert len(tokens["discard_token"]) == 2
    assert len(tokens["global_token"]) == 6
    assert all(len(t) == 6 for t in tokens["action_tokens"])
    assert len(tokens["decision_token"]) == 2


def test_worst_slot_prefers_high_known_then_unknown():
    env = HiddenHandEnv(seed=2)
    me = env.current_player
    hand = env.state.hands[me]
    hand.cards = [1, 7, 3, 2]
    hand.known_by = [[p == me for p in range(2)] if i in (0, 1) else [False, False] for i in range(4)]
    assert env.worst_slot(me) == 1  # known 7 beats unknown (4.5)
    hand.cards[1] = 4
    assert env.worst_slot(me) == 2  # all known <= 4 -> first unknown
    assert env.observe(me)["own_worst_slot"] == 2


def test_keep_discards_publicly_and_trade_moves_knowledge():
    env = HiddenHandEnv(seed=11)
    me = env.current_player
    other = 1 - me
    slot = env.worst_slot(me)
    old = env.state.hands[me].cards[slot]
    drawn = env.state.drawn
    env.step(KEEP_ACTION)
    assert env.state.hands[me].cards[slot] == drawn
    assert env.state.discard_pile[-1] == old
    assert env.observe(other)["discard_top"] == old
    assert env.state.hands[me].known_by[slot] == [me == 0, me == 1]
    # Opponent trades their worst card for my card at `slot`: they now hold my known card
    # at their worst position, and I know it (knowledge follows the card).
    their_draw = env.state.drawn
    their_slot = env.worst_slot(other)
    env.step(TRADE_BASE + slot)
    assert env.state.discard_pile[-1] == their_draw  # the unused draw is discarded face-up
    assert env.state.hands[other].cards[their_slot] == drawn
    assert env.state.hands[other].known_by[their_slot][me]
    assert not env.state.hands[other].known_by[their_slot][other]
    assert env.observe(me)["board_views"][other]["values"][their_slot] == drawn
    token = env.observe(me)["tokens"]["action_tokens"][-1]
    assert token[1] == 3 and token[2] == their_slot and token[3] == slot


def test_caller_penalty_and_winner():
    cfg = HiddenHandConfig(min_turns_before_call=0)
    env = HiddenHandEnv(seed=0, config=cfg)
    caller = env.current_player
    env.state.hands[caller].cards = [9, 9, 9, 9]
    env.state.hands[1 - caller].cards = [0, 0, 0, 0]
    _, rewards, done, info = env.step(CALL_ACTION)
    assert done
    assert env.scores[caller] == 72 and env.scores[1 - caller] == 0
    assert env.winner == 1 - caller
    # Default own-score reward: each player receives minus their own final score.
    assert rewards[f"player_{caller}"] == -72 and rewards[f"player_{1 - caller}"] == 0
    assert info["hidden"]["own_total"] == 36


@pytest.mark.parametrize("mode", ["own_score", "shaped", "score_diff", "win_loss"])
def test_reward_modes_sum_to_outcome(mode):
    for seed in range(100):
        env = HiddenHandEnv(seed=seed, config=HiddenHandConfig(reward_mode=mode))
        start = [h.total() for h in env.state.hands]
        rng = random.Random(seed)
        total = [0.0, 0.0]
        while not env.game_over:
            _, rewards, _, _ = env.step(rng.choice(env.legal_actions()))
            total = [total[p] + rewards[f"player_{p}"] for p in (0, 1)]
        final_diff = env.scores[1] - env.scores[0]
        if mode == "own_score":
            assert total == [pytest.approx(-env.scores[0]), pytest.approx(-env.scores[1])]
            continue
        expected = {
            "shaped": final_diff - (start[1] - start[0]),
            "score_diff": final_diff,
            "win_loss": {0: 1.0, 1: -1.0, -1: 0.0}[env.winner],
        }[mode]
        assert total[0] == pytest.approx(expected)
        assert total[1] == pytest.approx(-expected)


def test_hidden_features_channel():
    env = HiddenHandEnv(seed=5)
    feats = env.hidden_features(0)
    assert feats["opp_total"] == sum(env.state.hands[1].cards)
    assert feats["total_diff"] == feats["opp_total"] - feats["own_total"]


@pytest.mark.parametrize("name", ["random", "greedy", "tell_reader"])
def test_bots_only_play_legal_moves(name):
    for seed in range(100):
        env = HiddenHandEnv(seed=seed)
        bots = [make_bot(name, seed=seed), make_bot(name, seed=seed + 1)]
        while not env.game_over:
            legal = env.legal_actions()
            action = bots[env.current_player].select_action(env.observe(env.current_player), legal)
            assert action in legal
            env.step(action)


def test_action_space_and_mask():
    env = HiddenHandEnv(seed=1)
    mask = env.legal_action_mask()
    assert len(mask) == ACTION_SPACE_SIZE == 7
    assert [a for a, m in enumerate(mask) if m] == env.legal_actions()


def test_default_draw_pile_is_high_cards_only():
    env = HiddenHandEnv(seed=9)
    assert len(env.state.deck) + 1 == 20  # 4 copies of 5..9, one already drawn
    assert all(5 <= c <= 9 for c in env.state.deck) and 5 <= env.state.drawn <= 9
    dealt = [c for h in env.state.hands for c in h.cards]
    assert all(0 <= c <= 9 for c in dealt)
    legacy = HiddenHandEnv(seed=9, config=HiddenHandConfig(draw_pile_values=None))
    assert len(legacy.state.deck) + 1 == 32
