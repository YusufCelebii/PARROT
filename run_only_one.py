from __future__ import annotations
import json
import random
from typing import Optional, Dict, Any
from runners.config import DATASET_DEFAULT
from dotenv import load_dotenv
from runners.runner import run_one

DATASET_PATH = DATASET_DEFAULT

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)


INDEX: int = random.randint(1, total_lines)

def inspect_one(index: int = INDEX,
                dataset_path: Optional[str] = DATASET_PATH) -> Dict[str, Any]:
    """
    Programmatic API to inspect a single dataset item.
    Returns a dict with fields:
      - index, gold, asserted
      - base{answer, correct, probs, conf, explanation, raw, labels_seen_count}
      - manip{...}
      - metrics{changed, follow, scenario, delta_conf}
      - prompt_sent_to_model{base, manip}
    """
    return run_one(index, dataset_path=dataset_path)

def main() -> None:
    load_dotenv()
    result = inspect_one(INDEX, DATASET_PATH)

    print("\n=== INSPECT RESULT ===")
    print(f"PROMPT (BASE):\n{result['prompt_sent_to_model']['base']}\n")
    print(f"PROMPT (MANIP):\n{result['prompt_sent_to_model']['manip']}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
