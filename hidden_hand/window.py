"""
The game window for ``python -m hidden_hand.play --vis``, drawn with tkinter
(part of the standard library) from ``vis.table_view``.

Bots and agents move on a timer you can pause, single-step and slow down.
Humans move with the buttons, by clicking an opponent card to trade for it,
or with the keys k (keep), p (peek), 0-3 (trade), c (call).
Space pauses, Right arrow / n steps, Esc closes the window.
"""
from __future__ import annotations

import random
import tkinter as tk
import traceback
from tkinter import font as tkfont
from typing import Any, Callable, Sequence

from hidden_hand.env import CALL_ACTION, HAND_SIZE, KEEP_ACTION, PEEK_ACTION, TRADE_BASE, HiddenHandEnv, action_name
from hidden_hand.vis import table_view, tell_truths

FELT, FELT_DARK, PANEL = "#1e5038", "#143726", "#0f2a1d"
INK, MUTED = "#f4f1e8", "#9fbcab"
FACE, BACK, BACK_LINE = "#fbf8ef", "#2c4f86", "#7d9bd0"
LOW, HIGH = "#1e8449", "#c0392b"
LOW_TEXT, HIGH_TEXT = "#5fd68f", "#ff7a6b"
GOLD, DRAWN = "#f5c542", "#62d2f0"
PLAYER = ("#5bc0eb", "#f28dc0")

CARD_W, CARD_H = 72, 100
CANVAS_W, CANVAS_H = 680, 570
SLOT_X = [60, 152, 244, 336]
DRAW_X = 480
TOP_Y, MIDDLE_Y, BOTTOM_Y = 22, 212, 352


class GameWindow:
    """Plays ``seeds`` games between two players (from ``play.py``) and draws every move."""

    def __init__(
        self,
        players: Sequence[Any],
        seeds: Sequence[int],
        delay: float = 1.0,
        view_mode: str = "auto",
        paused: bool = False,
        on_game_over: Callable[[int, HiddenHandEnv], None] | None = None,
    ) -> None:
        self.players = list(players)
        self.names = [p.name for p in players]
        self.humans = [seat for seat, p in enumerate(players) if p.is_human]
        self.seeds = list(seeds)
        self.view_mode = view_mode
        self.paused = paused
        self.on_game_over = on_game_over
        self.tally = [0, 0, 0]  # P0 wins, P1 wins, draws
        self.illegal = 0
        self.game_index = -1
        self.env: HiddenHandEnv | None = None
        self.truths: list[bool | None] = []
        self.viewer: int | None = None  # with two humans: whose cards are on screen
        self.handoff = False
        self.note = ""
        self._job: str | None = None

        self.root = tk.Tk()
        self.root.title("Hidden Hand")
        self.root.configure(bg=FELT_DARK)
        self.root.resizable(False, False)
        self.delay = tk.DoubleVar(value=delay)
        self._fonts()
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _fonts(self) -> None:
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.f_head = tkfont.Font(family=family, size=15, weight="bold")
        self.f_text = tkfont.Font(family=family, size=12)
        self.f_bold = tkfont.Font(family=family, size=12, weight="bold")
        self.f_small = tkfont.Font(family=family, size=10)
        self.f_card = tkfont.Font(family=family, size=34, weight="bold")
        self.f_corner = tkfont.Font(family=family, size=11, weight="bold")
        self.f_banner = tkfont.Font(family=family, size=20, weight="bold")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy()
        self.f_mono.configure(size=12)

    def _build(self) -> None:
        root = self.root
        self.header = tk.Label(root, bg=FELT_DARK, fg=INK, font=self.f_bold, anchor="w")
        self.header.pack(fill="x", padx=12, pady=(10, 0))

        main = tk.Frame(root, bg=FELT_DARK)
        main.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(main, width=CANVAS_W, height=CANVAS_H, bg=FELT, highlightthickness=0)
        self.canvas.pack(side="left", padx=(12, 6), pady=8)
        side = tk.Frame(main, bg=FELT_DARK)
        side.pack(side="left", fill="both", expand=True, padx=(6, 12), pady=8)
        tk.Label(side, text="Public history", bg=FELT_DARK, fg=INK, font=self.f_bold, anchor="w").pack(fill="x")
        self.history = tk.Text(side, width=50, height=24, bg=PANEL, fg=INK, font=self.f_mono, wrap="word",
                               relief="flat", padx=8, pady=6, highlightthickness=0, cursor="arrow")
        self.history.pack(fill="both", expand=True, pady=(4, 0))
        for tag, opts in {
            "p0": {"foreground": PLAYER[0], "font": self._mono_bold()},
            "p1": {"foreground": PLAYER[1], "font": self._mono_bold()},
            "high": {"foreground": HIGH_TEXT}, "low": {"foreground": LOW_TEXT},
            "true": {"foreground": LOW_TEXT}, "false": {"foreground": GOLD},
            "dim": {"foreground": MUTED}, "newest": {"background": "#28583f"},
        }.items():
            self.history.tag_configure(tag, **opts)
        self.legend = tk.Label(side, bg=FELT_DARK, fg=MUTED, font=self.f_small, justify="left", anchor="w",
                               wraplength=400)
        self.legend.pack(fill="x", pady=(6, 0))

        bar = tk.Frame(root, bg=FELT_DARK)
        bar.pack(fill="x", padx=12, pady=(0, 12))
        self.status = tk.Label(bar, bg=FELT_DARK, fg=INK, font=self.f_bold, anchor="w")
        self.status.pack(fill="x")
        row = tk.Frame(bar, bg=FELT_DARK)
        row.pack(fill="x", pady=(6, 0))
        self.moves = tk.Frame(row, bg=FELT_DARK)
        self.moves.pack(side="left")
        controls = tk.Frame(row, bg=FELT_DARK)
        controls.pack(side="right")
        button = {"highlightbackground": FELT_DARK, "font": self.f_text}
        self.pause_button = tk.Button(controls, text="Resume" if self.paused else "Pause", width=7,
                                      command=self.toggle_pause, **button)
        self.step_button = tk.Button(controls, text="Step ▸", command=self.step_once, **button)
        self.next_button = tk.Button(controls, text="Next game", command=self.next_game, state="disabled", **button)
        tk.Label(controls, text="delay (s)", bg=FELT_DARK, fg=MUTED, font=self.f_small).pack(side="left")
        tk.Scale(controls, from_=0.0, to=3.0, resolution=0.1, orient="horizontal", length=110, variable=self.delay,
                 bg=FELT_DARK, fg=INK, troughcolor=PANEL, highlightthickness=0, font=self.f_small).pack(side="left")
        for b in (self.pause_button, self.step_button, self.next_button):
            b.pack(side="left", padx=(6, 0))

        for j in range(HAND_SIZE):
            self.canvas.tag_bind(f"trade{j}", "<Button-1>", lambda _e, j=j: self.human_move(TRADE_BASE + j))
            self.canvas.tag_bind(f"trade{j}", "<Enter>", lambda _e: self.canvas.config(cursor="hand2"))
            self.canvas.tag_bind(f"trade{j}", "<Leave>", lambda _e: self.canvas.config(cursor=""))
        self.canvas.tag_bind("handoff", "<Button-1>", lambda _e: self.end_handoff())
        root.bind("<Key>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _mono_bold(self) -> tkfont.Font:
        f = self.f_mono.copy()
        f.configure(weight="bold")
        return f

    # ------------------------------------------------------------------
    # Running games
    # ------------------------------------------------------------------
    def run(self) -> list[int]:
        self.root.after(50, self.next_game)
        self.root.mainloop()
        return self.tally

    def close(self) -> None:
        self._cancel()
        self.root.destroy()

    def next_game(self) -> None:
        self._cancel()
        if self.game_index + 1 >= len(self.seeds):
            return
        self.game_index += 1
        self.env = HiddenHandEnv(seed=self.seeds[self.game_index])
        self.truths = tell_truths(self.env, [])
        self.viewer = None
        self.note = ""
        self.next_button.config(state="disabled")
        self._turn()

    def _turn(self) -> None:
        """Show the decision that is up: a human waits for input, a bot moves after the delay."""
        seat = self.env.current_player
        if self.players[seat].is_human and len(self.humans) == 2 and self.view_mode == "auto":
            self.handoff = self.viewer != seat
        self.redraw()
        self._continue()

    def _continue(self) -> None:
        env = self.env
        if self.paused or self.handoff:
            return
        if env.game_over:
            if self.game_index + 1 < len(self.seeds) and not self.humans:
                self._schedule(max(1.5, 3 * self.delay.get()), self.next_game)
        elif not self.players[env.current_player].is_human:
            self._schedule(self.delay.get(), self.step_once)

    def step_once(self) -> None:
        """Play one bot/agent move now (or start the next game if this one is over)."""
        self._cancel()
        env = self.env
        if env is None or self.handoff:
            return
        if env.game_over:
            self.next_game()
            return
        seat = env.current_player
        player = self.players[seat]
        if player.is_human:
            return
        self.status.config(text=f"P{seat} {player.name} is thinking…")
        self.root.update_idletasks()
        legal = env.legal_actions()
        try:
            action = player.choose(env.observe(seat), legal)
        except Exception:
            traceback.print_exc()
            self.paused = True
            self.pause_button.config(text="Resume")
            self.status.config(text=f"P{seat} {player.name} raised an error (traceback in the terminal)", fg=GOLD)
            return
        self.note = ""
        if action not in legal:
            self.illegal += 1
            self.note = f"P{seat} chose illegal {action_name(action)}, replaced at random. "
            action = random.Random(self.seeds[self.game_index] + env.state.step).choice(legal)
        self._apply(action)

    def human_move(self, action: int) -> None:
        env = self.env
        if env is None or env.game_over or self.handoff or not self.players[env.current_player].is_human:
            return
        if action not in env.legal_actions():
            self.root.bell()
            return
        self.note = ""
        self._apply(action)

    def _apply(self, action: int) -> None:
        env = self.env
        env.step(action)
        tell_truths(env, self.truths)
        if env.game_over:
            self.tally[env.winner] += 1
            if self.on_game_over:
                self.on_game_over(self.game_index, env)
            if self.game_index + 1 < len(self.seeds):
                self.next_button.config(state="normal")
            self.redraw()
            self._continue()
        else:
            self._turn()

    def end_handoff(self) -> None:
        if self.handoff:
            self.handoff = False
            self.viewer = self.env.current_player
            self.redraw()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        if self.paused:
            self._cancel()
        else:
            self._continue()
        if self.env is not None:
            self._draw_status(self._model())

    def _schedule(self, seconds: float, fn: Callable[[], None]) -> None:
        self._cancel()
        self._job = self.root.after(int(seconds * 1000), fn)

    def _cancel(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None

    def _on_key(self, event: tk.Event) -> None:
        key = (event.char or "").lower()
        if event.keysym == "Escape":
            self.close()
        elif self.handoff and event.keysym in ("Return", "space"):
            self.end_handoff()
        elif key == " ":
            self.toggle_pause()
        elif event.keysym == "Right" or key == "n":
            self.step_once()
        elif key in ("k", "p", "c"):
            self.human_move({"k": KEEP_ACTION, "p": PEEK_ACTION, "c": CALL_ACTION}[key])
        elif key.isdigit() and int(key) < HAND_SIZE:
            self.human_move(TRADE_BASE + int(key))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def current_view(self) -> int | None:
        if self.view_mode == "all":
            return None
        if self.view_mode in ("0", "1"):
            return int(self.view_mode)
        if len(self.humans) == 1:
            return self.humans[0]
        if len(self.humans) == 2:
            return self.viewer if self.viewer is not None else self.env.current_player
        return None

    def _model(self) -> dict:
        return table_view(self.env, self.current_view(), self.names, self.truths)

    def _human_turn(self) -> bool:
        env = self.env
        return not env.game_over and not self.handoff and self.players[env.current_player].is_human

    def redraw(self) -> None:
        model = self._model()
        view = model["view"]
        self.root.title(f"Hidden Hand · P0 {self.names[0]} vs P1 {self.names[1]}")
        self.header.config(text=f"Hidden Hand   game {self.game_index + 1}/{len(self.seeds)} · "
                                f"seed {self.seeds[self.game_index]} · step {model['step']}/{model['max_steps']}   "
                                + ("spectator view" if view is None else f"P{view}'s view"))
        self._draw_table(model)
        self._draw_history(model)
        self._draw_moves(model)
        self._draw_status(model)
        self.legend.config(text="Gold border: changed by the last move.  ▲ worst: the card KEEP and TRADE replace."
                                + ("  Chips under a card: who knows it.  ✓ / ✗: the tell was right / wrong."
                                   if model["reveal"] else "  ░ cards are ones you have not seen."))

    def _draw_status(self, model: dict) -> None:
        env = self.env
        result = model["result"]
        if result is not None:
            outcome = "Draw" if result["winner"] == -1 else f"P{result['winner']} {self.names[result['winner']]} wins"
            text = f"{outcome} · {result['scores'][0]} : {result['scores'][1]}"
            p0, p1, draws = self.tally
            text += f"   ·   tally P0 {p0} · P1 {p1} · draws {draws}"
            if self.game_index + 1 >= len(self.seeds):
                text += "   ·   all games played, close the window to exit"
            elif self.humans or self.paused:
                text += "   ·   press Next game"
        elif self.handoff:
            text = f"Pass the screen to P{env.current_player}"
        else:
            seat = env.current_player
            player = self.players[seat]
            if player.is_human:
                drawn = f"  You drew: {model['drawn']}" if model["drawn"] is not None else ""
                text = f"P{seat}, your move.{drawn}"
            else:
                text = f"P{seat} {player.name} to move" + ("  (paused: Step ▸ or Right arrow)" if self.paused else "")
        self.status.config(text=self.note + text, fg=GOLD if self.note else INK)

    def _draw_moves(self, model: dict) -> None:
        for child in self.moves.winfo_children():
            child.destroy()
        if not self._human_turn():
            return
        env = self.env
        me = env.current_player
        legal = env.legal_actions()
        worst = env.worst_slot(me)
        hand = env.state.hands[me]
        button = {"highlightbackground": FELT_DARK, "font": self.f_text}
        if KEEP_ACTION in legal:
            tk.Button(self.moves, text=f"Keep {env.state.drawn} → slot {worst}  (k)",
                      command=lambda: self.human_move(KEEP_ACTION), **button).pack(side="left", padx=(0, 6))
        if PEEK_ACTION in legal:
            pos = next(i for i in range(HAND_SIZE) if not hand.known_by[i][me])
            tk.Button(self.moves, text=f"Peek slot {pos}  (p)",
                      command=lambda: self.human_move(PEEK_ACTION), **button).pack(side="left", padx=(0, 6))
        for j in range(HAND_SIZE):
            if TRADE_BASE + j in legal:
                tk.Button(self.moves, text=f"Trade for {j}", command=lambda j=j: self.human_move(TRADE_BASE + j),
                          **button).pack(side="left", padx=(0, 4))
        if CALL_ACTION in legal:
            tk.Button(self.moves, text="Call  (c)", command=lambda: self.human_move(CALL_ACTION),
                      **button).pack(side="left", padx=(6, 0))

    def _draw_history(self, model: dict) -> None:
        h = self.history
        h.config(state="normal")
        h.delete("1.0", "end")
        if not self.handoff:
            for entry in model["history"]:
                start = h.index("end-1c")
                h.insert("end", f"{'deal' if entry['step'] == 0 else entry['step']:>4}  ", ("dim",))
                for text, tag in entry["parts"]:
                    h.insert("end", text, (tag,) if tag else ())
                h.insert("end", "\n")
                if entry["newest"]:
                    h.tag_add("newest", start, "end-1c")
        h.config(state="disabled")
        h.see("end")

    def _draw_table(self, model: dict) -> None:
        c = self.canvas
        c.delete("all")
        if self.handoff:
            seat = self.env.current_player
            c.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill=FELT, outline="", tags=("handoff",))
            c.create_text(CANVAS_W / 2, CANVAS_H / 2 - 20, text=f"Pass the screen to P{seat}",
                          fill=PLAYER[seat], font=self.f_banner, tags=("handoff",))
            c.create_text(CANVAS_W / 2, CANVAS_H / 2 + 16, text="click here or press Enter when ready",
                          fill=MUTED, font=self.f_text, tags=("handoff",))
            return
        view = model["view"]
        top, bottom = (1, 0) if view in (None, 0) else (0, 1)
        trade_clicks = self._human_turn()
        self._draw_hand(model, model["hands"][top], TOP_Y, trade_clicks and top != self.env.current_player)
        self._draw_piles(model)
        self._draw_hand(model, model["hands"][bottom], BOTTOM_Y, trade_clicks and bottom != self.env.current_player)
        if model["result"] is not None:
            self._draw_result(model)

    def _draw_hand(self, model: dict, hand: dict, y0: int, trade_clicks: bool) -> None:
        c = self.canvas
        owner = hand["owner"]
        to_move = model["to_move"] == owner
        label = ("▶  " if to_move else "") + f"P{owner}  {hand['name']}"
        if owner == model["view"] and model["result"] is None:
            label += "  (you)"
        c.create_text(24, y0, text=label, anchor="w", fill=PLAYER[owner], font=self.f_head)
        info = f"turn {hand['turns']}/{model['max_turns']}   ·   "
        if hand["total"] is not None:
            info += f"total {hand['total']}"
        else:
            info += f"seen cards sum {hand['seen_sum']}   ·   {hand['unseen']} unseen"
        c.create_text(CANVAS_W - 24, y0, text=info, anchor="e", fill=MUTED, font=self.f_text)

        y = y0 + 20
        for pos, card in enumerate(hand["cards"]):
            x = SLOT_X[pos]
            tags = (f"trade{pos}",) if trade_clicks else ()
            self._card(x, y, card["value"], model["high_threshold"], changed=card["changed"], tags=tags)
            c.create_text(x + CARD_W / 2, y + CARD_H + 12, text=f"slot {pos}", fill=MUTED, font=self.f_small)
            if card["seen_by"] is not None:
                self._chips(x + CARD_W / 2, y + CARD_H + 31, card["seen_by"])
            if card["worst"]:
                c.create_text(x + CARD_W / 2, y + CARD_H + 50, text="▲ worst", fill=GOLD, font=self.f_corner)
        if trade_clicks:
            c.create_text(SLOT_X[-1] + CARD_W + 24, y + CARD_H / 2, anchor="w", fill=MUTED, font=self.f_small,
                          text="click a card\nto trade for it")
        if to_move and model["has_draw"]:
            self._card(DRAW_X, y, model["drawn"], model["high_threshold"], drawn=True)
            c.create_text(DRAW_X + CARD_W / 2, y + CARD_H + 12, text="drawn card", fill=DRAWN, font=self.f_small)

    def _draw_piles(self, model: dict) -> None:
        c = self.canvas
        y = MIDDLE_Y
        x = 110
        for k in (2, 1, 0) if model["deck_size"] else ():
            self._card(x + 3 * k, y - 3 * k, None, model["high_threshold"], label="")
        c.create_text(x + CARD_W / 2, y + CARD_H + 14, text=f"draw pile · {model['deck_size']}",
                      fill=MUTED, font=self.f_small)
        x = 240
        discards = model["discards"]
        if discards:
            self._card(x, y, discards[-1], model["high_threshold"])
        else:
            self._rrect(x, y, x + CARD_W, y + CARD_H, 9, fill="", outline=MUTED, dash=(4, 3))
        c.create_text(x + CARD_W / 2, y + CARD_H + 14, text="discard pile", fill=MUTED, font=self.f_small)
        earlier = discards[:-1][-9:]
        if earlier:
            x0 = 350
            c.create_text(x0, y + 18, text="earlier discards", anchor="w", fill=MUTED, font=self.f_small)
            for i, value in enumerate(earlier):
                cx = x0 + 30 * i
                self._rrect(cx, y + 34, cx + 25, y + 68, 4, fill=FACE, outline="#0b2418")
                color = HIGH if value >= model["high_threshold"] else LOW
                c.create_text(cx + 12.5, y + 51, text=str(value), fill=color, font=self.f_corner)

    def _draw_result(self, model: dict) -> None:
        c = self.canvas
        r = model["result"]
        x1, y1, x2, y2 = 80, MIDDLE_Y - 8, CANVAS_W - 80, MIDDLE_Y + CARD_H + 26
        self._rrect(x1, y1, x2, y2, 14, fill=PANEL, outline=GOLD, width=2)
        if r["winner"] == -1:
            title, color = "Draw", GOLD
        else:
            title, color = f"P{r['winner']} {self.names[r['winner']]} wins", PLAYER[r["winner"]]
        c.create_text(CANVAS_W / 2, y1 + 26, text=title, fill=color, font=self.f_banner)
        how = f"P{r['caller']} called" if r["caller"] >= 0 else "all turns played"
        c.create_text(CANVAS_W / 2, y1 + 52, text=how, fill=MUTED, font=self.f_text)
        for p in range(2):
            line = f"P{p} {self.names[p]}:  total {r['totals'][p]}  →  score {r['scores'][p]}"
            if p == r["caller"] and r["doubled"]:
                line += "  (caller not strictly lowest: doubled)"
            c.create_text(CANVAS_W / 2, y1 + 78 + 20 * p, text=line, fill=INK, font=self.f_text)

    # ------------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------------
    def _rrect(self, x1: float, y1: float, x2: float, y2: float, r: float, **kw: Any) -> int:
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _card(self, x: float, y: float, value: int | None, high: int, changed: bool = False, drawn: bool = False,
              tags: tuple[str, ...] = (), label: str = "?") -> None:
        c = self.canvas
        outline, width = "#0b2418", 1
        if drawn:
            outline, width = DRAWN, 3
        if changed:
            outline, width = GOLD, 4
        if value is None:
            self._rrect(x, y, x + CARD_W, y + CARD_H, 9, fill=BACK, outline=outline, width=width, tags=tags)
            self._rrect(x + 7, y + 7, x + CARD_W - 7, y + CARD_H - 7, 6, fill="", outline=BACK_LINE, tags=tags)
            if label:
                c.create_text(x + CARD_W / 2, y + CARD_H / 2, text=label, fill="#c9d6ee", font=self.f_card, tags=tags)
            return
        color = HIGH if value >= high else LOW
        self._rrect(x, y, x + CARD_W, y + CARD_H, 9, fill=FACE, outline=outline, width=width, tags=tags)
        c.create_text(x + CARD_W / 2, y + CARD_H / 2, text=str(value), fill=color, font=self.f_card, tags=tags)
        c.create_text(x + 10, y + 12, text=str(value), fill=color, font=self.f_corner, tags=tags)
        c.create_text(x + CARD_W - 10, y + CARD_H - 12, text=str(value), fill=color, font=self.f_corner, tags=tags)

    def _chips(self, cx: float, y: float, seen_by: list[int]) -> None:
        c = self.canvas
        if not seen_by:
            c.create_text(cx, y, text="nobody", fill=MUTED, font=self.f_small)
            return
        x = cx - 14 * len(seen_by)
        for p in seen_by:
            self._rrect(x + 1, y - 8, x + 27, y + 8, 6, fill=PLAYER[p], outline="")
            c.create_text(x + 14, y, text=f"P{p}", fill=PANEL, font=self.f_corner)
            x += 28
