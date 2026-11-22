"""Central configuration for the Fake News BERT project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "WELFake_Dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models" / "best_model"
METRICS_DIR = ROOT / "outputs" / "metrics"

BERT_NAME = "bert-base-uncased"
RANDOM_SEED = 42
TEST_SIZE = 0.15          # held-out test set, never used for model selection
VAL_SIZE = 0.15           # validation set, used for early stopping / checkpoint choice
MAX_LENGTH = 256          # tokens; lower (128) for faster CPU training
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 1              # epochs without val-loss improvement before stopping
SAMPLE_SIZE = None        # e.g. 20000 for quick CPU runs; None = full dataset
MAX_BATCH_ROWS = 2000     # cap on rows processed from a batch CSV upload

LABEL_MAP = {0: "FAKE", 1: "REAL"}
