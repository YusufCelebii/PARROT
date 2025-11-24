# runners/logprobs.py
from __future__ import annotations
import math
import re
from typing import Dict, List, Optional

_PUNCT_SIDES = " ()[]{}:;.,'\""


# --------- small utils ---------
def _as_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return str(x)
    return str(x)


def _strip_side_punct(x: str) -> str:
    s = (x or "")
    if s and s[0] in _PUNCT_SIDES:
        s = s[1:].lstrip()
    if s and s[-1] in _PUNCT_SIDES:
        s = s[:-1].rstrip()
    return s


def canonical_label(tok: str, labels_set) -> str | None:
    x = _as_str(tok).strip()
    x = _strip_side_punct(x)
    x = x.strip().upper()
    return x if (len(x) == 1 and x in labels_set) else None


def _norm_word(tok: str) -> str:
    x = _as_str(tok).strip().lower()
    x = _strip_side_punct(x)
    return x


def softmax_from_logps(logp_map: Dict[str, float]) -> Dict[str, float]:
    if not logp_map:
        return {}
    m = max(logp_map.values())
    exps = {k: math.exp(v - m) for k, v in logp_map.items()}
    s = sum(exps.values()) or 1.0
    return {k: (v / s) for k, v in exps.items()}


def softmax_from_logps_tau(logp_map: Dict[str, float], tau: float = 1.0) -> Dict[str, float]:
    if not logp_map:
        return {}
    t = max(float(tau), 1e-6)
    m = max(logp_map.values())
    exps = {k: math.exp((v - m) / t) for k, v in logp_map.items()}
    s = sum(exps.values()) or 1.0
    return {k: (v / s) for k, v in exps.items()}


def logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


# --------- pooling over top-k ---------
def _dedupe_by_token(top_list_at_pos) -> List[Dict[str, float]]:
    best: Dict[str, float] = {}
    for item in (top_list_at_pos or []):
        tok = _as_str(item.get("token"))
        lp = item.get("logprob")
        if not isinstance(lp, (int, float)):
            continue
        if tok in best:
            if float(lp) > best[tok]:
                best[tok] = float(lp)
        else:
            best[tok] = float(lp)
    return [{"token": t, "logprob": lp} for t, lp in best.items()]


def build_label_masses(top_list_at_pos, labels: List[str], pool: str = "max") -> Dict[str, float]:
    labels_set = set(labels)
    buckets: Dict[str, List[float]] = {L: [] for L in labels}
    for item in _dedupe_by_token(top_list_at_pos):
        lab = canonical_label(item.get("token"), labels_set)
        if lab:
            buckets[lab].append(float(item.get("logprob")))
    out: Dict[str, float] = {}
    for L, lps in buckets.items():
        if not lps:
            continue
        if pool == "lse":
            out[L] = logsumexp(lps)
        else:
            out[L] = max(lps)
    return out


# --------- anchored position ---------
def select_final_anchored_position(tokens: List[str], labels: List[str]) -> int | None:
    labels_set = set(labels)
    N = len(tokens or [])
    for i in range(N):
        if _norm_word(tokens[i]) == "final":
            for j in range(i + 1, min(i + 13, N)):
                if canonical_label(tokens[j], labels_set):
                    return j
    return None


# --------- main API ---------
def label_probs_from_logprobs_anchored(
    meta_logprobs: Optional[dict],
    labels: List[str],
    pool_mode: str = "max",
    tau: float = 1.0,
) -> Dict[str, float]:
    if not meta_logprobs:
        return {}

    toks = meta_logprobs.get("tokens") or []
    tops = meta_logprobs.get("top_logprobs") or []

    pos = select_final_anchored_position(toks, labels)
    if pos is None or pos >= len(tops):
        best_idx, best_score = None, None
        for i in range(min(len(toks), len(tops))):
            masses_i = build_label_masses(tops[i], labels, pool=pool_mode)
            if not masses_i:
                continue
            score = logsumexp(list(masses_i.values()))
            if best_score is None or score > best_score:
                best_score, best_idx = score, i
        pos = best_idx

    if pos is None or pos >= len(tops):
        return {}

    masses = build_label_masses(tops[pos], labels, pool=pool_mode)

    observed = [float(v) for v in masses.values()]
    if observed:
        min_lp = min(observed)
        tiny = min_lp - 5.0
    else:
        tiny = -25.0

    log_map = {L: masses.get(L, tiny) for L in labels}
    return softmax_from_logps_tau(log_map, tau)


def prob_of_letter(prob_map: Dict[str, float], letter: Optional[str]) -> Optional[float]:
    if not prob_map or not letter:
        return None
    key = letter.strip().upper()
    return float(prob_map[key]) if key in prob_map else None


def labels_seen_count(meta_lp: Optional[dict], labels: List[str]) -> int:
    if not meta_lp:
        return 0
    toks = meta_lp.get("tokens") or []
    tops = meta_lp.get("top_logprobs") or []

    pos = select_final_anchored_position(toks, labels)
    if pos is None or pos >= len(tops):
        return 0

    label_set = {L.upper() for L in labels}
    seen = 0
    for item in _dedupe_by_token(tops[pos]):
        tok = (_as_str(item.get("token")) or "").strip().upper()
        if tok in label_set:
            seen += 1
    return seen


# --------- DEBUG HELPERS ---------
def reconstruct_text(meta_logprobs: Optional[dict]) -> str:
    if not meta_logprobs:
        return ""
    toks = meta_logprobs.get("tokens") or []
    return "".join(_as_str(t) for t in toks)


def render_with_anchor_mark(meta_logprobs: Optional[dict], pos: Optional[int], *, window: int = 0) -> str:
    if not meta_logprobs or pos is None:
        return ""
    toks = meta_logprobs.get("tokens") or []
    N = len(toks)
    if pos < 0 or pos >= N:
        return ""
    left = "".join(_as_str(t) for t in toks[:pos])
    mid  = _as_str(toks[pos])
    right = "".join(_as_str(t) for t in toks[pos+1:])
    return left + "<ANCHOR>" + mid + "</ANCHOR>" + right


def anchor_debug(meta_logprobs: Optional[dict], labels: List[str], *, window: int = 3, max_top: int = 10) -> Dict[str, object]:
        
    out = {
        "anchored_pos": None,
        "left_tokens": [],
        "chosen_token": None,
        "right_tokens": [],
        "chosen_token_logprob": None,
        "top_logprobs_at_pos": [],
        "reconstructed_anchor_slice": "",
    }
    if not meta_logprobs:
        return out

    toks = meta_logprobs.get("tokens") or []
    tops = meta_logprobs.get("top_logprobs") or []
    tlps = meta_logprobs.get("token_logprobs") or []

    pos = select_final_anchored_position(toks, labels)
    if pos is None or pos >= len(tops):
        best_idx, best_score = None, None
        for i in range(min(len(toks), len(tops))):
            masses_i = build_label_masses(tops[i], labels, pool="max")
            if not masses_i:
                continue
            score = logsumexp(list(masses_i.values()))
            if best_score is None or score > best_score:
                best_score, best_idx = score, i
        pos = best_idx

    out["anchored_pos"] = pos
    if pos is None:
        return out

    L = max(0, pos - window)
    R = min(len(toks), pos + 1 + window)
    out["left_tokens"]  = [_as_str(t) for t in toks[L:pos]]
    out["chosen_token"] = _as_str(toks[pos]) if pos < len(toks) else None
    out["right_tokens"] = [_as_str(t) for t in toks[pos+1:R]]

    out["chosen_token_logprob"] = float(tlps[pos]) if (pos is not None and pos < len(tlps) and isinstance(tlps[pos], (int, float))) else None

    top_at = tops[pos] if (pos is not None and pos < len(tops)) else []
    top_dump = []
    for itm in (top_at or [])[:max_top]:
        tok = _as_str(itm.get("token"))
        lp = itm.get("logprob")
        top_dump.append({"token": tok, "logprob": float(lp) if isinstance(lp, (int, float)) else None})
    out["top_logprobs_at_pos"] = top_dump

    out["reconstructed_anchor_slice"] = render_with_anchor_mark(meta_logprobs, pos, window=0)
    return out


__all__ = [
    "label_probs_from_logprobs_anchored",
    "prob_of_letter",
    "labels_seen_count",
    "select_final_anchored_position",
    "build_label_masses",
    "canonical_label",
    "softmax_from_logps",
    "softmax_from_logps_tau",
    # debug helpers
    "reconstruct_text",
    "render_with_anchor_mark",
    "anchor_debug",
]
