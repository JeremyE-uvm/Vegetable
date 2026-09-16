"""
Core game logic for Veggie Semantle, kept separate from the GUI so it
can be tested or reused (e.g. in a CLI or web version) independently.
"""
import random
from dataclasses import dataclass, field


@dataclass
class Guess:
    word: str
    similarity: float
    rank: int
    total: int
    point: tuple
    is_answer: bool


@dataclass
class GameState:
    answer: str = ""
    answer_point: tuple = (0.0, 0.0)
    guesses: list = field(default_factory=list)
    won: bool = False
    gave_up: bool = False


class VeggieSemantle:
    """
    Wraps an EmbeddingStore with per-game state: the secret vegetable
    and the running list of guesses.
    """

    def __init__(self, embedding_store, seed=None):
        self.store = embedding_store
        self.rng = random.Random(seed)
        self.state = GameState()
        self._answer_vec = None
        self.new_game()

    def new_game(self):
        answer = self.rng.choice(self.store.vegetables)
        answer_vec = self.store.embed_word(answer)
        answer_point = self.store.project(answer_vec)
        self._answer_vec = answer_vec
        self.state = GameState(answer=answer, answer_point=answer_point)

    def guess(self, word):
        if self.state.gave_up:
            return None

        word_clean = word.strip().lower()
        if not word_clean:
            return None

        vec = self.store.embed_word(word_clean)
        sim = self.store.cosine_similarity(vec, self._answer_vec)
        rank, total = self.store.rank_against_database(vec, self._answer_vec)
        point = self.store.project(vec)
        is_answer = word_clean == self.state.answer

        g = Guess(
            word=word_clean,
            similarity=sim,
            rank=rank,
            total=total,
            point=point,
            is_answer=is_answer,
        )
        self.state.guesses.append(g)
        if is_answer:
            self.state.won = True
        return g
