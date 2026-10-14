"""Tests for the game runner (python -m hidden_hand.play) and the window's table model."""
from __future__ import annotations

import copy
import random

import pytest

from hidden_hand import CALL_ACTION, KEEP_ACTION, PEEK_ACTION, TRADE_BASE, HiddenHandEnv
from hidden_hand.play import QuitGame, main, make_player, parse_move
from hidden_hand.vis import table_view, tell_truths


def test_player_view_never_shows_hidden_cards():
    # Rewriting every card the viewer does not know (and a draw they cannot see) must not
    # change anything the window would draw for them.
    for seed in range(40):
        env = HiddenHandEnv(seed=seed)
        truths = tell_truths(env, [])
        rng = random.Random(seed)
        while not env.game_over:
            for viewer in (0, 1):
                before = table_view(env, viewer, truths=truths)
                saved = copy.deepcopy([h.cards for h in env.state.hands]), env.state.drawn
                for hand in env.state.hands:
                    for pos, kb in enumerate(hand.known_by):
                        if not kb[viewer]:
                            hand.cards[pos] = (hand.cards[pos] + 3) % 10
                if viewer != env.current_player:
                    env.state.drawn = (env.state.drawn + 1) % 10
                assert table_view(env, viewer, truths=truths) == before
                for hand, cards in zip(env.state.hands, saved[0]):
                    hand.cards[:] = cards
                env.state.drawn = saved[1]
            env.step(rng.choice(env.legal_actions()))
            tell_truths(env, truths)


def test_spectator_and_game_over_reveal_everything():
    env = HiddenHandEnv(seed=2)
    spectator = table_view(env)
    assert all(c["value"] is not None for h in spectator["hands"] for c in h["cards"])
    assert spectator["drawn"] == env.state.drawn
    rng = random.Random(2)
    while not env.game_over:
        env.step(rng.choice(env.legal_actions()))
    final = table_view(env, view=0)
    assert [[c["value"] for c in h["cards"]] for h in final["hands"]] == [h.cards for h in env.state.hands]
    assert final["result"]["scores"] == env.scores


def test_tell_truths_track_every_public_action():
    env = HiddenHandEnv(seed=4)
    truths = tell_truths(env, [])
    rng = random.Random(4)
    while not env.game_over:
        env.step(rng.choice(env.legal_actions()))
        tell_truths(env, truths)
    assert len(truths) == len(env.state.history)
    assert all((t is None) == (a.tell == -127) for t, a in zip(truths, env.state.history))


def test_parse_move():
    legal = [KEEP_ACTION, PEEK_ACTION, TRADE_BASE, TRADE_BASE + 3]
    assert parse_move("k", legal) == KEEP_ACTION
    assert parse_move(" Peek ", legal) == PEEK_ACTION
    assert parse_move("3", legal) == TRADE_BASE + 3
    assert parse_move("trade 0", legal) == TRADE_BASE
    assert parse_move("c", legal) is None  # CALL not legal here
    assert parse_move("c", legal + [CALL_ACTION]) == CALL_ACTION
    assert parse_move("7", legal) is None
    with pytest.raises(QuitGame):
        parse_move("q", legal)


def test_text_mode_bots(capsys):
    main(["greedy", "tell_reader", "--games", "3"])
    out = capsys.readouterr().out
    assert out.count("scores") == 3 and "3 game(s)" in out


def test_text_mode_human(monkeypatch, capsys):
    moves = iter(["nonsense", "p", "0", "c", "c", "c", "c", "c", "c"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(moves))
    main(["human", "random", "--seed", "1"])
    out = capsys.readouterr().out
    assert "Not a legal move" in out and "1 game(s)" in out


@pytest.fixture
def window_factory():
    tk = pytest.importorskip("tkinter")
    from hidden_hand.window import GameWindow

    opened = []

    def make(specs, games=1, **kwargs):
        try:
            w = GameWindow([make_player(s, i) for i, s in enumerate(specs)], list(range(games)),
                           delay=0, paused=True, **kwargs)
        except tk.TclError as exc:
            pytest.skip(f"no display: {exc}")
        w.root.withdraw()
        opened.append(w)
        return w

    yield make
    for w in opened:
        w.close()


def test_window_plays_bots_to_the_end(window_factory):
    finished = []
    w = window_factory(["greedy", "tell_reader"], games=2, on_game_over=lambda i, env: finished.append(i))
    w.next_game()
    assert w.current_view() is None
    while len(finished) < 2:
        w.step_once()  # also moves on to the next game
    assert finished == [0, 1] and sum(w.tally) == 2
    assert "all games played" in w.status.cget("text")


def test_window_human_moves_and_hides_opponent(window_factory):
    w = window_factory(["human", "tell_reader"])
    w.next_game()
    assert w.current_view() == 0
    while not w.env.game_over:
        if w.env.current_player == 0:
            assert w.moves.winfo_children()  # move buttons are up
            w.human_move(CALL_ACTION if CALL_ACTION in w.env.legal_actions() else KEEP_ACTION)
        else:
            w.human_move(KEEP_ACTION)  # ignored: not the human's turn
            assert w.env.current_player == 1
            w.step_once()
    assert sum(w.tally) == 1


def test_window_hot_seat_hands_off(window_factory):
    w = window_factory(["human", "human"])
    w.next_game()
    assert w.handoff
    w.human_move(w.env.legal_actions()[0])  # ignored until the hand-off is confirmed
    assert w.env.state.step == 0
    w.end_handoff()
    seat = w.env.current_player
    assert w.current_view() == seat
    w.human_move(PEEK_ACTION)
    assert w.handoff and w.env.current_player == 1 - seat
