# runners/config.py
from __future__ import annotations
import os
from datetime import datetime

# ============== Core run config ==============
PROVIDER = os.getenv("PROVIDER", "openai").lower()
MODEL    = os.getenv("MODEL", "gpt-3.5-turbo").lower()
ALPHABET = list("ABCD")

# Logprobs / pooling
WANT_LOGPROBS = int(os.getenv("WANT_LOGPROBS", "19"))
# Default pooling for label mass aggregation at the anchored position.
# "max" | "lse"
POOL_MODE = os.getenv("POOL_MODE", "lse").strip().lower()

# ===== Temperature scaling (τ) for calibrated class probabilities =====
# Applied when converting label log-masses ccinto probabilities.
# τ=1.0 => no change; τ>1 softens; τ<1 sharpens. Argmax unchanged.
TAU = float(os.getenv("TAU", "3.0"))

# Sampling
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P       = float(os.getenv("TOP_P", "1.0"))

# Repro
SEED = int(os.getenv("SEED", "42"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from pathlib import Path
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
DATASET_DEFAULT = str(PROJECT_ROOT / "datasets" / "mmlu-ALL.jsonl")

def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
