"""Project paths (everything is relative to the repository root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_TXT = DATA / "GPAC.txt"
ARTIFACTS = ROOT / "artifacts"
CORPUS_RAW = ARTIFACTS / "corpus_raw"          # encoded, before de-duplication
CORPUS_CLEAN = ARTIFACTS / "corpus_clean"      # encoded, de-duplicated
SPLITS = ARTIFACTS / "splits.npz"
MODEL_FULL = ARTIFACTS / "models" / "full"     # all clean data, used by Task 1 & 4
MODEL_TRAIN = ARTIFACTS / "models" / "train"   # 90% split, <UNK> for rare words; Task 2 & 3
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
FONT = ROOT / "fonts" / "NotoSansEthiopic.ttf"
GDRIVE_ID = "1jEOAORylTBAiNKes_JerD6gFF7cnLFyr"
