# utils/io_utils.py
from __future__ import annotations
import os, csv, json
from typing import List, Dict, Any

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


CORE_FIELDS = [
    "index", "subset", "gold", "asserted",
    "base_ans", "mani_ans",
    "base_probs", "mani_probs",
    "question", "base_expl", "mani_expl",
]

def _coerce_json_str(row: Dict[str, Any]) -> Dict[str, Any]:
    """base_probs / mani_probs alanlarını JSON string'e zorla."""
    out = dict(row)
    if not isinstance(out.get("base_probs"), str):
        out["base_probs"] = json.dumps(out.get("base_probs") or {}, ensure_ascii=False)
    if not isinstance(out.get("mani_probs"), str):
        out["mani_probs"] = json.dumps(out.get("mani_probs") or {}, ensure_ascii=False)
    return out

def write_core_csv(path: str, rows: List[Dict]):
    """(İSİM KORUNDU) Artık minimal primitive şemayı yazar."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CORE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(_coerce_json_str(r))

