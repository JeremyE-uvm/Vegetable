"""
Veggie Semantle -- entry point.

Run this to launch the game:
    python main.py

First launch will download the sentence-transformers model and embed
the vegetable database (cached to embedding_cache.npz afterward, so
subsequent launches start much faster).
"""
from gui import VeggieSemantleApp

if __name__ == "__main__":
    app = VeggieSemantleApp()
    app.mainloop()
