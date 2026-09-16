"""
Tkinter GUI for Veggie Semantle.

Shows a guess entry box, a running history table (sorted by closeness),
and a live 2D UMAP scatter plot of every guess made so far.
"""
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from embeddings import EmbeddingStore
from game import VeggieSemantle


class VeggieSemantleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Veggie Semantle")
        self.geometry("980x640")
        self.minsize(820, 560)

        self.status_var = tk.StringVar(value="Loading model and vegetable database...")
        self.guess_var = tk.StringVar()

        self._build_layout()

        # Load the model/data after the window is visible, so it doesn't
        # look frozen while the (one-time, then cached) embeddings build.
        self.after(50, self._init_game)

    # ---------- layout ----------

    def _build_layout(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(
            top, textvariable=self.status_var, font=("Segoe UI", 10, "italic"), wraplength=940
        ).pack(side=tk.TOP, anchor="w")

        entry_row = ttk.Frame(self, padding=(10, 0, 10, 10))
        entry_row.pack(side=tk.TOP, fill=tk.X)

        self.entry = ttk.Entry(entry_row, textvariable=self.guess_var, font=("Segoe UI", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda e: self._submit_guess())
        self.entry.state(["disabled"])

        self.submit_btn = ttk.Button(entry_row, text="Guess", command=self._submit_guess)
        self.submit_btn.pack(side=tk.LEFT)
        self.submit_btn.state(["disabled"])

        self.new_game_btn = ttk.Button(entry_row, text="New Game", command=self._new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.new_game_btn.state(["disabled"])

        self.give_up_btn = ttk.Button(entry_row, text="Give Up", command=self._give_up)
        self.give_up_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.give_up_btn.state(["disabled"])

        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: history table
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("guess", "similarity", "rank")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=20)
        self.tree.heading("guess", text="Guess")
        self.tree.heading("similarity", text="Cosine Sim")
        self.tree.heading("rank", text="Rank (of DB)")
        self.tree.column("guess", width=160, anchor="w")
        self.tree.column("similarity", width=110, anchor="center")
        self.tree.column("rank", width=110, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.LEFT, fill=tk.Y)

        # Right: UMAP plot
        right = ttk.Frame(body, width=420)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 0))

        self.fig = Figure(figsize=(4.6, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Guesses in UMAP space")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- game lifecycle ----------

    def _init_game(self):
        try:
            self.store = EmbeddingStore()
            self.game = VeggieSemantle(self.store)
        except Exception as e:
            messagebox.showerror("Startup error", f"Failed to load model/data:\n{e}")
            self.status_var.set("Failed to start. See error dialog.")
            return

        self.status_var.set(
            f"Guess a vegetable! Database has {len(self.store.vegetables)} possible answers."
        )
        self.entry.state(["!disabled"])
        self.submit_btn.state(["!disabled"])
        self.new_game_btn.state(["!disabled"])
        self.give_up_btn.state(["!disabled"])
        self.entry.focus_set()
        self._redraw_plot()

    def _new_game(self):
        self.game.new_game()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.status_var.set(
            f"New game started. Guess a vegetable! Database has "
            f"{len(self.store.vegetables)} possible answers."
        )
        self.guess_var.set("")
        self.entry.state(["!disabled"])
        self.submit_btn.state(["!disabled"])
        self.give_up_btn.state(["!disabled"])
        self._redraw_plot()
        self.entry.focus_set()

    def _give_up(self):
        if self.game is None or self.game.state.won or self.game.state.gave_up:
            return

        self.game.state.gave_up = True
        self.status_var.set(
            f"You gave up. The vegetable was '{self.game.state.answer}'. "
            "Click New Game to play again."
        )
        self.entry.state(["disabled"])
        self.submit_btn.state(["disabled"])
        self.give_up_btn.state(["disabled"])
        self._redraw_plot()

    # ---------- guessing ----------

    def _submit_guess(self):
        word = self.guess_var.get()
        if not word.strip() or self.game.state.won or self.game.state.gave_up:
            return

        g = self.game.guess(word)
        self.guess_var.set("")
        if g is None:
            return

        self._insert_history_row(g)
        self._redraw_plot()

        if g.is_answer:
            self.status_var.set(
                f"Correct! The vegetable was '{self.game.state.answer}'. "
                f"Solved in {len(self.game.state.guesses)} guesses. Click New Game to play again."
            )
            self.entry.state(["disabled"])
            self.submit_btn.state(["disabled"])
            self.give_up_btn.state(["disabled"])
        else:
            self.status_var.set(
                f"'{g.word}' -- cosine similarity {g.similarity:.4f}, "
                f"rank {g.rank} of {g.total} in the vegetable database."
            )

    def _insert_history_row(self, g):
        self.tree.insert(
            "", "end", values=(g.word, f"{g.similarity:.4f}", f"{g.rank}/{g.total}")
        )
        # Keep the table sorted by similarity, closest guess on top.
        rows = [(self.tree.set(k, "similarity"), k) for k in self.tree.get_children("")]
        rows.sort(key=lambda r: float(r[0]), reverse=True)
        for index, (_, k) in enumerate(rows):
            self.tree.move(k, "", index)

    # ---------- plotting ----------

    def _redraw_plot(self):
        self.ax.clear()
        self.ax.set_title("Guesses in UMAP space")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        xs = [g.point[0] for g in self.game.state.guesses]
        ys = [g.point[1] for g in self.game.state.guesses]
        if xs:
            self.ax.scatter(xs, ys, c="steelblue", s=40, zorder=2)
            for g in self.game.state.guesses:
                self.ax.annotate(
                    g.word, g.point, fontsize=8, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points",
                )

        if self.game.state.won or self.game.state.gave_up:
            ax_pt = self.game.state.answer_point
            self.ax.scatter([ax_pt[0]], [ax_pt[1]], c="crimson", s=90, marker="*", zorder=3)
            self.ax.annotate(
                self.game.state.answer, ax_pt, fontsize=9, weight="bold",
                color="crimson", xytext=(4, 4), textcoords="offset points",
            )

        self.canvas.draw()


if __name__ == "__main__":
    app = VeggieSemantleApp()
    app.mainloop()
