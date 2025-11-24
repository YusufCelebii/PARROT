# mmlu_dataset.py
from __future__ import annotations
from typing import List, Dict
from datasets import load_dataset
import string, random, os, json

def _letters(n: int) -> str:
    return string.ascii_uppercase[:n]

def _answers_block(choices: List[str]) -> str:
    L = _letters(len(choices))
    return "\n".join([f"({L[i]}){choices[i]}" for i in range(len(choices))])

def _prompt_obj(question: str, answers_str: str) -> List[Dict[str, str]]:
    human = f"{question}\n{answers_str}\nPlease answer just with the letter of the correct answer."
    return [
        {"type": "human", "content": human},
        {"type": "ai", "content": "The answer is ("}
    ]

def _row_to_template(ex: Dict, subject: str) -> Dict:
    q: str = ex["question"]
    choices: List[str] = ex["choices"]
    gold_idx: int = int(ex["answer"])
    L = _letters(len(choices))
    answers_str = _answers_block(choices)
    return {
        "prompt": _prompt_obj(q, answers_str),
        "base": {
            "dataset": f"mmlu/{subject}",
            "question": q,
            "correct_letter": L[gold_idx],
            "answers": answers_str
        },
        "metadata": {
            "prompt_template": "{question}\n{answers}\nPlease answer just with the letter of the correct answer."
        }
    }

def load_mmlu_dataset(
    subject: str = "professional_law",
    split: str = "test",
    n: int = 50,
    seed: int = 42,
    shuffle: bool = False,
) -> List[Dict]:
    ds = load_dataset("cais/mmlu", subject, split="test")
    idx = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    idx = idx[: min(n, len(idx))]
    subset = ds.select(idx)
    return [_row_to_template(ex, subject) for ex in subset]

def save_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    subject = "computer_security"
    n = 3
    rows = load_mmlu_dataset(subject=subject, n=n, shuffle=True, seed=42)
    out_path = os.path.join("datasets", f"mmlu-_{subject}_{n}.jsonl")
    save_jsonl(rows, out_path)
    print(f"Saved {len(rows)} rows to {out_path}")
