"""
Embedding utilities for Veggie Semantle.

Loads a Sentence-Transformers model, builds/caches embeddings for the
vegetable answer database (from database.csv), and fits a UMAP
projection on that fixed database so every guess -- and the secret
answer -- can be projected into the same consistent 2D space.
"""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap

MODEL_NAME = "all-mpnet-base-v2"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
VEGETABLE_CSV = os.path.join(DATA_DIR, "database.csv")
CACHE_PATH = os.path.join(DATA_DIR, "embedding_cache.npz")


class EmbeddingStore:
    """
    Owns the sentence-transformer model, the vegetable database, its
    cached embeddings, and the PCA projection fit on that database.
    """

    def __init__(self, csv_path=VEGETABLE_CSV, cache_path=CACHE_PATH):
        self.model = SentenceTransformer(MODEL_NAME)
        self.vegetables = self._load_vegetables(csv_path)
        self.db_embeddings = self._load_or_build_embeddings(cache_path)

        # Fit UMAP once on the database. This is the fixed coordinate
        # system every guess (and the answer) gets projected into, so
        # the plot's axes don't shift as you play.
        self.reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
        self.db_points = self.reducer.fit_transform(self.db_embeddings)

    # ---------- loading ----------

    def _load_vegetables(self, csv_path):
        df = pd.read_csv(csv_path)

        # The CSV includes an unnamed index column. Use the actual vegetable names
        # column when present, otherwise fall back to the first non-index column.
        veggie_candidates = [
            c for c in df.columns
            if str(c).strip().lower() == "vegetable" or str(c).strip().startswith("unnamed")
        ]
        if "Vegetable" in df.columns:
            col = "Vegetable"
        elif len(df.columns) > 1:
            col = [c for c in df.columns if str(c).strip().lower() != "embedding" and not str(c).strip().startswith("unnamed")][0]
        else:
            col = df.columns[0]

        words = [str(w).strip().lower() for w in df[col].tolist() if str(w).strip()]
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        return unique_words

    def _load_or_build_embeddings(self, cache_path):
        if os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            cached_words = list(cached["words"])
            if cached_words == self.vegetables:
                return cached["embeddings"]
        embeddings = self.model.encode(self.vegetables, normalize_embeddings=True)
        np.savez(
            cache_path,
            words=np.array(self.vegetables, dtype=object),
            embeddings=embeddings,
        )
        return embeddings

    # ---------- runtime use ----------

    def embed_word(self, word):
        """Embed an arbitrary guess word (normalized, so dot product = cosine sim)."""
        vec = self.model.encode([word.strip().lower()], normalize_embeddings=True)[0]
        return vec

    def project(self, vec):
        """Project a single embedding into the fixed 2D UMAP space."""
        point = self.reducer.transform(vec.reshape(1, -1))[0]
        return float(point[0]), float(point[1])

    def cosine_similarity(self, vec_a, vec_b):
        # Embeddings are L2-normalized at encode time, so dot product IS cosine similarity.
        return float(np.dot(vec_a, vec_b))

    def rank_against_database(self, guess_vec, answer_vec):
        """
        Rank the guess's similarity-to-answer against every vegetable in
        the database's similarity-to-answer. Rank 1 = as close as the
        closest possible database word.

        Note: since guesses can be *any* word, there's no huge fixed
        vocabulary to rank against (unlike the original Semantle). This
        ranks the guess against the ~60-word vegetable database instead,
        which is the answer pool.
        """
        guess_sim = self.cosine_similarity(guess_vec, answer_vec)
        db_sims = self.db_embeddings @ answer_vec
        rank = int(np.sum(db_sims > guess_sim)) + 1
        return rank, len(self.vegetables)
