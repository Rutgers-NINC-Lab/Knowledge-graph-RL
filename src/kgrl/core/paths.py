"""Project path constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
EMBEDDING_ROOT = PROJECT_ROOT / "embeddings"
CHECKPOINT_ROOT = PROJECT_ROOT / "models" / "checkpoints"
