from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Iterator, Tuple

# proje kökünü sys.path'e ekle
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.config import ALPHABET, DATASET_DEFAULT

def iter_dataset(path: str) -> Iterator[Tuple[str, str, str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            obj = json.loads(line)
            base = obj.get("base") or {}
            q = base.get("question")
            a = base.get("answers")
            gold = (base.get("correct_letter") or "").upper()
            subset = (base.get("subset") or "").strip()  # <-- EKLENDİ
            if not subset:
                ds = (base.get("dataset") or "").strip()
                # "mmlu/subject" -> "subject"
                subset = ds.split("/", 1)[1] if ("/" in ds) else ds

            if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                yield q.strip(), a.strip(), gold, subset

def get_nth_item(path: str, n: int) -> Tuple[str, str, str, str]:
    if n <= 0:
        raise ValueError("n must be >= 1 (1-based index)")
    for i, quad in enumerate(iter_dataset(path), start=1):
        if i == n:
            return quad
    raise IndexError(f"Dataset has fewer than {n} items.")

if __name__ == "__main__":
    row = get_nth_item(DATASET_DEFAULT, 1)
    print({
        "question": row[0],
        "answers": row[1],
        "correct_letter": row[2]
    })
