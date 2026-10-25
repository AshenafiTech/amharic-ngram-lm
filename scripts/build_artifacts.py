"""Build every cached artifact: encoded corpus, splits, full and train models.

    python scripts/build_artifacts.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amharic_lm import pipeline  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    clean = pipeline.clean_corpus()
    print(f"clean corpus: {clean.n_sentences:,} sentences, {clean.n_tokens:,} tokens")
    pipeline.splits(clean)
    del clean
    print("training full model (orders 1-4) ...")
    pipeline.full_model()
    print("training train-split model (orders 1-4, min_count=2) ...")
    pipeline.train_model()
    print(f"done in {time.time() - t0:.0f}s")
