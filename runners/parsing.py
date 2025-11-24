from __future__ import annotations
import json
import re
from typing import Tuple, Optional


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove first fence
        s = s[s.find("\n")+1:] if "\n" in s else s
        # remove trailing fence if present
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()

def parse_output(text: str) -> Tuple[Optional[str], str]:
    """
    Return (final_letter, explanation_text). Tolerates:
      - code fences like ```json ... ```
      - stray prose around the JSON
    """
    if not text:
        return None, ""

    candidate = _strip_code_fence(text)

    # If still contains prose, extract the first JSON object substring.
    if not candidate.strip().startswith("{"):
        m = _JSON_OBJ_RE.search(candidate)
        if m:
            candidate = m.group(0)

    final, explanation = None, ""
    try:
        obj = json.loads(candidate)
        final = (obj.get("final") or "").strip() or None
        if final:
            final = final.upper()
        explanation = (obj.get("explanation") or "").strip()
    except Exception:
        # Best-effort fallback: try to spot `"final": "X"`
        m = re.search(r'"final"\s*:\s*"([A-Za-z])"', candidate)
        if m:
            final = m.group(1).upper()
        m2 = re.search(r'"explanation"\s*:\s*"([\s\S]*?)"\s*[,}]', candidate)
        if m2:
            explanation = m2.group(1).strip()

    return final, explanation
