# runners/runner.py
from __future__ import annotations
import os, random, time, json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from .prompts import SYS, USER_TPL, get_manip_suffix
from .config import (
    PROVIDER, MODEL, ALPHABET, DATASET_DEFAULT,
    TEMPERATURE, TOP_P, WANT_LOGPROBS, SEED, stamp, POOL_MODE, TAU
)
from .parsing import parse_output
from .logprobs import (
    label_probs_from_logprobs_anchored,
    prob_of_letter,
    labels_seen_count,
    anchor_debug,
    render_with_anchor_mark,
)
from .metrics import round6, guidance_label, scenario_of


from utils.io_utils import ensure_dir, write_core_csv
from utils.dataset_reader import iter_dataset, get_nth_item

from llm_client import LLMClient


def _sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths by replacing invalid characters."""
    return model.replace("/", "_").replace("\\", "_").replace(":", "_")


def _ask_with_retry(
    cl,
    prompt: str,
    system: str,
    want_logprobs: int,
    initial_sleep: float = 1.5,
    backoff: float = 1.6,
    max_sleep: float = 20.0,
):
    import time, random
    attempt, sleep_s = 0, max(0.0, float(initial_sleep))
    while True:
        try:
            res = cl.ask(
                prompt,
                system=system,
                want_logprobs=want_logprobs,
                retry_forever=True,
                initial_sleep=initial_sleep,
                backoff=backoff,
                max_sleep=max_sleep,
            )
            raw = (res["text"] if isinstance(res, dict) else (res or ""))
            meta_lp = (res.get("meta", {}).get("logprobs") if isinstance(res, dict) else None)
            probs = (label_probs_from_logprobs_anchored(meta_lp, ALPHABET, pool_mode=POOL_MODE, tau=TAU) if meta_lp else {})
            letter, expl = parse_output(raw)
            if (not isinstance(raw, str)) or (not raw.strip()) or (not letter):
                raise RuntimeError("empty_or_unparsed_response")
            return raw, letter, expl, probs, meta_lp
        except Exception as e:
            attempt += 1
            wait = sleep_s + random.uniform(0.0, sleep_s * 0.25)
            print(f"parse retry {attempt}: {e} -> {wait:.1f}s")
            time.sleep(wait)
            sleep_s = min(max_sleep, max(0.5, sleep_s * backoff))

def _print_anchor_debug(tag: str, meta_lp, labels):
    try:
        dbg = anchor_debug(meta_lp, labels, window=4, max_top=10)
        print(f"[ANCHOR DEBUG - {tag}]")
        print(json.dumps(dbg, ensure_ascii=False, indent=2))
       
        print(f"[ANCHOR TEXT  - {tag}]")
        print((render_with_anchor_mark(meta_lp, dbg.get("anchored_pos"), window=0) or "")[:400])
    except Exception as _e:
    
        print(f"[ANCHOR DEBUG - {tag}] <unavailable: {type(_e).__name__}: {_e}>")

def run_one(n: int, dataset_path: Optional[str] = None) -> Dict:
    load_dotenv()
    ds_path = dataset_path or DATASET_DEFAULT
    cl = LLMClient(PROVIDER, MODEL, temperature=TEMPERATURE, top_p=TOP_P)

    q, a, gold, subset = get_nth_item(ds_path, n)
    base_user = USER_TPL.format(q=q, a=a, labels="|".join(ALPHABET))

    try:
        raw_b, ans_b, expl_b, probs_b, meta_b = _ask_with_retry(cl, base_user, SYS, WANT_LOGPROBS)
    except Exception as e:
        raw_b, ans_b, expl_b, probs_b, meta_b = (str(e), "", "", {}, None)
    base_conf = round6(prob_of_letter(probs_b, ans_b))

    # ---- DEBUG: BASE
    _print_anchor_debug("BASE", meta_b, ALPHABET)

    wrong_choices = [L for L in ALPHABET if L != gold]
    rng = random.Random(SEED + int(n))
    asserted = rng.choice(wrong_choices) if wrong_choices else gold
    manip_user = base_user + get_manip_suffix(subset, asserted)

    try:
        raw_m, ans_m, expl_m, probs_m, meta_m = _ask_with_retry(cl, manip_user, SYS, WANT_LOGPROBS)
    except Exception as e:
        raw_m, ans_m, expl_m, probs_m, meta_m = (str(e), "", "", {}, None)
    mani_conf = round6(prob_of_letter(probs_m, ans_m))

    # ---- DEBUG: MANIP
    _print_anchor_debug("MANIP", meta_m, ALPHABET)

    delta_conf = None
    if base_conf is not None and mani_conf is not None:
        delta_conf = round6(mani_conf - base_conf)

    base_correct = int(ans_b == gold if ans_b else False)
    mani_correct = int(ans_m == gold if ans_m else False)
    guidance = guidance_label(asserted, gold)
    changed = int((ans_m or "") != (ans_b or ""))
    follow = int((ans_m or "") == asserted)
    scenario = scenario_of(base_correct, guidance, changed)

    p_gold_b   = (probs_b or {}).get(gold)
    p_assert_b = (probs_b or {}).get(asserted)
    p_gold_m   = (probs_m or {}).get(gold)
    p_assert_m = (probs_m or {}).get(asserted)

    delta_p_gold   = round6((p_gold_m or 0.0) - (p_gold_b or 0.0))
    delta_p_assert = round6((p_assert_m or 0.0) - (p_assert_b or 0.0))

    return {
        "index": n,
        "subset": subset,
        "gold": gold,
        "asserted": asserted,
        "base": {
            "answer": ans_b, "correct": base_correct, "probs": probs_b, "conf": base_conf,
            "explanation": expl_b, "raw": raw_b, "labels_seen_count": labels_seen_count(meta_b, ALPHABET),
        },
        "manip": {
            "answer": ans_m, "correct": mani_correct, "probs": probs_m, "conf": mani_conf,
            "explanation": expl_m, "raw": raw_m, "labels_seen_count": labels_seen_count(meta_m, ALPHABET),
        },
        "metrics": {
            "changed": changed,
            "follow": follow,
            "scenario": scenario,
            "delta_conf": delta_conf,
            "p_gold_base": round6(p_gold_b),
            "p_asserted_base": round6(p_assert_b),
            "p_gold_mani": round6(p_gold_m),
            "p_asserted_mani": round6(p_assert_m),
            "delta_p_gold": delta_p_gold,
            "delta_p_asserted": delta_p_assert,
        },
        "prompt_sent_to_model": {
        "base": base_user,              
        "manip": manip_user             
},
    }


def run_all(dataset_path: Optional[str] = None,
            out_dir: Optional[str] = None,
            max_samples: Optional[int] = None):
    load_dotenv()
    ds_path = dataset_path or DATASET_DEFAULT
    cl = LLMClient(PROVIDER, MODEL, temperature=TEMPERATURE, top_p=TOP_P)

    ts = stamp()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Sanitize model name for file paths
    safe_model = _sanitize_model_name(MODEL)
    run_dir = ensure_dir(out_dir or os.path.join(base_dir, "outputs", f"{PROVIDER}_{safe_model}_{ts}"))
    core_csv = os.path.join(run_dir, f"core_results_{PROVIDER}_{safe_model}_{ts}.csv")


    run_id = f"{PROVIDER}_{safe_model}_{ts}"
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta = {
        "run_id": run_id,
        "timestamp": ts,
        "dataset_path": ds_path,
        "provider": PROVIDER,
        "model": MODEL,
        "config": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "tau": TAU,
            "pool_mode": POOL_MODE,
            "want_logprobs": WANT_LOGPROBS,
            "seed": SEED,
        },
        "env": {
            "python": f"{__import__('sys').version.split()[0]}",
            "platform": __import__("platform").platform(),
        },
    }

    rows_core: List[Dict] = []
    n_total = n_correct_base = n_correct_mani = n_follow = 0
    t0 = time.time()

    
    for idx, (q, a, gold, subset) in enumerate(iter_dataset(ds_path), start=1):
        if max_samples and idx > max_samples:
            break

        n_total += 1
        base_user = USER_TPL.format(q=q, a=a, labels="|".join(ALPHABET))

        try:
            raw_b, ans_b, expl_b, probs_b, meta_b = _ask_with_retry(cl, base_user, SYS, WANT_LOGPROBS)
        except Exception as e:
            raw_b, ans_b, expl_b, probs_b, meta_b = (str(e), "", "", {}, None)

        
        _print_anchor_debug(f"BASE#{idx}", meta_b, ALPHABET)

        wrong_choices = [L for L in ALPHABET if L != gold]
        rng = random.Random(SEED + int(idx))
        asserted = rng.choice(wrong_choices) if wrong_choices else gold

        manip_user = base_user + get_manip_suffix(subset, asserted)


        try:
            raw_m, ans_m, expl_m, probs_m, meta_m = _ask_with_retry(cl, manip_user, SYS, WANT_LOGPROBS)
        except Exception as e:
            raw_m, ans_m, expl_m, probs_m, meta_m = (str(e), "", "", {}, None)

        # ---- DEBUG: MANIP
        _print_anchor_debug(f"MANIP#{idx}", meta_m, ALPHABET)

        base_correct = int(ans_b == gold if ans_b else False)
        mani_correct = int(ans_m == gold if ans_m else False)
        if base_correct: n_correct_base += 1
        if mani_correct: n_correct_mani += 1
        if (ans_m or "") == asserted: n_follow += 1

        # ---- SADE (primitive-only) satır — 
        q_full = " ".join((q or "").split())
        row_min = {
            "index": idx,
            "subset": subset,
            "gold": gold,
            "asserted": asserted,
            "base_ans": ans_b,
            "mani_ans": ans_m,
            "base_probs": probs_b or {},
            "mani_probs": probs_m or {},
            "question": q_full,
            "base_expl": expl_b,
            "mani_expl": expl_m,
        }

        rows_core.append(row_min)

        print(f"[{idx}] elapsed={time.time()-t0:.1f}s "
              f"base={ans_b} mani={ans_m} follow={int((ans_m or '') == asserted)}")

  
    write_core_csv(core_csv, rows_core)

    total = max(n_total, 1)

    meta["summary"] = {
    "samples": n_total,
    "base_acc": n_correct_base / total,
    "mani_acc": n_correct_mani / total,
    "follow_rate": n_follow / total,
    "core_csv": core_csv,
}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    
    
    return {
        "summary": {
            "provider": PROVIDER, "model": MODEL,
            "samples": n_total,
            "base_acc": n_correct_base / total,
            "mani_acc": n_correct_mani / total,
            "follow_rate": n_follow / total,
            "core_csv": core_csv,
        }
    }
