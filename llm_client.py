# llm_client.py
from __future__ import annotations
import os
from typing import Optional, Sequence, Dict, Any, Protocol, List, Union

# -------------------------
# Provider protocol & registry
# -------------------------
class Provider(Protocol):
    def call(
        self,
        *,
        model: str,
        system: Optional[str],
        messages: Sequence[Dict[str, Any]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]: ...
    def list_models(self, timeout: Optional[int]) -> List[str]: ...

_PROVIDERS: Dict[str, Provider] = {}

def register_provider(name: str):
    name = name.lower().strip()
    def _wrap(cls):
        _PROVIDERS[name] = cls()
        return cls
    return _wrap

# -------------------------
# Core client
# -------------------------
class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout_s: Optional[int] = 30,
    ):
        p = provider.lower().strip()
        if p not in _PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider!r}")
        self._provider: Provider = _PROVIDERS[p]
        self.provider_name = p
        self.model = model
        self.temperature = 0.0 if temperature is None else float(temperature)
        self.top_p = 1.0 if top_p is None else float(top_p)
        self.timeout_s = timeout_s

    def ask(
    self,
    prompt: str,
    *,
    system: Optional[str] = None,
    want_logprobs: Union[bool, int] = False,
    retry_forever: bool = False,
    initial_sleep: float = 1.5,
    backoff: float = 1.6,
    max_sleep: float = 20.0,
) -> Union[str, Dict[str, Any]]:
        import time, random
        msgs: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        attempt, sleep_s = 0, max(0.0, float(initial_sleep))
        while True:
            try:
                res = self._provider.call(
                    model=self.model,
                    system=system,
                    messages=msgs,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    timeout=self.timeout_s,
                    want_logprobs=want_logprobs,
                )
                txt = (res.get("text", "").strip() if isinstance(res, dict) else (res or "").strip())
                if not txt:
                    raise RuntimeError("empty_response_text")
                return res
            except Exception as e:
                attempt += 1
                wait = sleep_s + random.uniform(0.0, sleep_s * 0.25)
                print(f"ask retry {attempt}: {e} -> {wait:.1f}s")
                time.sleep(wait)
                sleep_s = min(max_sleep, max(0.5, sleep_s * backoff))
                if (not retry_forever) and attempt >= 3:
                    return {"text": "", "meta": {"error": "max retries exceeded", "last_exc": str(e)}}

    def ask_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        return self._provider.call(
            model=self.model,
            system=None,
            messages=list(messages),
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout_s,
            want_logprobs=want_logprobs,
)


    def list_models(self) -> List[str]:
        return self._provider.list_models(self.timeout_s)

# -------------------------
# Helpers
# -------------------------
def _normalize_topk(top_list):
    out = []
    for x in top_list or []:
        tok = getattr(x, "token", None)
        lp = getattr(x, "logprob", None)
        if isinstance(x, dict):
            tok = x.get("token", tok)
            lp = x.get("logprob", lp)
        if isinstance(tok, (bytes, bytearray)):
            try:
                tok = tok.decode("utf-8", "ignore")
            except Exception:
                tok = str(tok)
        if isinstance(tok, str) and isinstance(lp, (int, float)):
            out.append({"token": tok, "logprob": float(lp)})
    return out

def _mk_meta(provider: str, model: str, finish_reason: Optional[str],
             logprobs_obj: Optional[Dict[str, Any]],
             warning: Optional[str] = None):
    return {
        "provider": provider,
        "model": model,
        "finish_reason": finish_reason,
        "logprobs": logprobs_obj,
        "warning": warning,
    }

def _maybe_logprobs_return(text: str, provider: str, model: str,
                           log: Optional[Dict[str, Any]],
                           want_k: int, finish: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """
    Unified wrapper for logprobs:
    - If want_k == 0 -> return plain text
    - If log exists   -> return dict with real logprobs
    - If not supported -> return dict with warning + logprobs=None
    """
    if not want_k:
        return text
    if log is None:
        return {"text": text,
                "meta": _mk_meta(provider, model, finish, None,
                                 "logprobs unsupported for this provider/model")}
    return {"text": text,
            "meta": _mk_meta(provider, model, finish, log, None)}

# -------------------------
# Providers
# -------------------------
@register_provider("vertex")
class VertexProvider:
    @staticmethod
    def _lp_to_simple(lp) -> Optional[Dict[str, Any]]:
        """Vertex LogprobsResult (proto) -> common dict: {tokens, token_logprobs, top_logprobs}"""
        try:
            chosen = getattr(lp, "chosen_candidates", []) or []
            tokens = [c.token for c in chosen]
            token_lps = [c.log_probability for c in chosen]

            top = []
            for pos in (getattr(lp, "top_candidates", []) or []):
                cands = getattr(pos, "candidates", []) or []
                top.append([{"token": c.token, "logprob": c.log_probability} for c in cands])

            return {"tokens": tokens, "token_logprobs": token_lps, "top_logprobs": top}
        except Exception:
            return None

    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        import os, vertexai
        from vertexai.generative_models import GenerativeModel

        project = os.getenv("VERTEX_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("VERTEX_PROJECT not set")

        vertexai.init(project=project, location=location)

        parts = []
        if system:
            parts.append(system.strip())
        for m in messages:
            c = (m.get("content") or "").strip()
            if c:
                parts.append(c)
        prompt = "\n\n".join(parts).strip()

        
        gen_cfg: Dict[str, Any] = {}
        gen_cfg["temperature"] = float(0.0 if temperature is None else temperature)
        gen_cfg["top_p"] = float(1.0 if top_p is None else top_p)
        gen_cfg["max_output_tokens"] = 256


        want_k = 5 if want_logprobs is True else (int(want_logprobs) if isinstance(want_logprobs, int) and want_logprobs > 0 else 0)
        if want_k:
            gen_cfg["response_logprobs"] = True
            gen_cfg["logprobs"] = want_k

        gm = GenerativeModel(model)
        resp = gm.generate_content(prompt, generation_config=gen_cfg)
        out_text = ""
        try:
            t = getattr(resp, "text", None)
            if isinstance(t, str) and t.strip():
                out_text = t.strip()
        except Exception:
            pass

        logprobs_obj = None
        if want_k:
            try:
                c0 = (getattr(resp, "candidates", None) or [None])[0]
                if c0 is not None:
                    raw_lp = getattr(c0, "logprobs_result", None) or getattr(c0, "logprobs", None)
                    if raw_lp is not None:
                        logprobs_obj = self._lp_to_simple(raw_lp)  # <-- ortak dict
            except Exception:
                logprobs_obj = None

        return _maybe_logprobs_return(out_text, "vertex", model, logprobs_obj, want_k, None)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        import os
        from google.auth import default
        from google.auth.transport.requests import AuthorizedSession
        from google.cloud import aiplatform_v1

        location = os.getenv("VERTEX_LOCATION", "us-central1")
        project = os.getenv("VERTEX_PROJECT")

        out: set[str] = set()

        # Publisher modelleri (REST v1beta1)
        try:
            endpoint = f"https://{location}-aiplatform.googleapis.com"
            url = f"{endpoint}/v1beta1/publishers/google/models"
            creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            session = AuthorizedSession(creds)
            page_token = None
            while True:
                params = {"pageSize": 200}
                if page_token:
                    params["pageToken"] = page_token
                r = session.get(url, params=params, timeout=timeout or 30)
                r.raise_for_status()
                data = r.json()
                for m in data.get("publisherModels", []):
                    rid = (m.get("name") or "").split("/")[-1]
                    if rid:
                        out.add(rid)
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        except Exception:
            pass

        try:
            if project:
                parent = f"projects/{project}/locations/{location}"
                ms = aiplatform_v1.ModelServiceClient()
                for mdl in ms.list_models(parent=parent):
                    rid = (getattr(mdl, "name", "") or "").split("/")[-1] or getattr(mdl, "display_name", None)
                    if rid:
                        out.add(rid)
        except Exception:
            pass

        return sorted(out) if out else []

@register_provider("openai")
class OpenAIProvider:
    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        from openai import OpenAI
        
        api = os.getenv("OPENAI_API_KEY")
        
        if not api:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api)

        final_msgs = []
        if system:
            final_msgs.append({"role": "system", "content": system})
        final_msgs.extend(messages)

        params: Dict[str, Any] = {"model": model, "messages": final_msgs}
        
        params["temperature"] = float(0.0 if temperature is None else temperature)
        params["top_p"] = float(1.0 if top_p is None else top_p)

        want_k = 5 if want_logprobs is True else (int(want_logprobs) if isinstance(want_logprobs, int) and want_logprobs > 0 else 0)
        if want_k:
            params["logprobs"] = True
            params["top_logprobs"] = want_k

        try:
            resp = client.chat.completions.create(**params)
        except Exception as e:
            if "not allowed to request logprobs" in str(e).lower():
                params.pop("logprobs", None)
                params.pop("top_logprobs", None)
                want_k = 0
                resp = client.chat.completions.create(**params)
            else:
                raise

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)

        logprobs_obj = None
        if want_k:
            log = getattr(choice, "logprobs", None)
            tokens, token_lps, top_by_pos = [], [], []
            if log and getattr(log, "content", None):
                for pos in log.content:
                    tok = getattr(pos, "token", None)
                    lp = getattr(pos, "logprob", None)
                    if isinstance(tok, str):
                        tokens.append(tok)
                    if isinstance(lp, (int, float)):
                        token_lps.append(float(lp))
                    top_by_pos.append(_normalize_topk(getattr(pos, "top_logprobs", None)))
                logprobs_obj = {
                    "tokens": tokens,
                    "token_logprobs": token_lps,
                    "top_logprobs": top_by_pos,
                }

        return _maybe_logprobs_return(text, "openai", model, logprobs_obj, want_k, finish)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        from openai import OpenAI
        api = os.getenv("OPENAI_API_KEY")
        if not api:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api)
        return sorted([m.id for m in client.models.list().data])


@register_provider("deepseek")
class DeepSeekProvider(OpenAIProvider):
    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        from openai import OpenAI
        api = os.getenv("DEEPSEEK_API_KEY")
        if not api:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        client = OpenAI(api_key=api, base_url="https://api.deepseek.com")

        final_msgs = []
        if system:
            final_msgs.append({"role": "system", "content": system})
        final_msgs.extend(messages)

        params: Dict[str, Any] = {"model": model, "messages": final_msgs}
        params["temperature"] = float(0.0 if temperature is None else temperature)
        params["top_p"] = float(1.0 if top_p is None else top_p)

        want_k = 5 if want_logprobs is True else (int(want_logprobs) if isinstance(want_logprobs, int) and want_logprobs > 0 else 0)
        if want_k:
            params["logprobs"] = True
            params["top_logprobs"] = want_k

        try:
            resp = client.chat.completions.create(**params)
        except Exception as e:
            if "not allowed to request logprobs" in str(e).lower():
                params.pop("logprobs", None)
                params.pop("top_logprobs", None)
                want_k = 0
                resp = client.chat.completions.create(**params)
            else:
                raise

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)

        logprobs_obj = None
        if want_k:
            log = getattr(choice, "logprobs", None)
            tokens, token_lps, top_by_pos = [], [], []
            if log and getattr(log, "content", None):
                for pos in log.content:
                    tok = getattr(pos, "token", None)
                    lp = getattr(pos, "logprob", None)
                    if isinstance(tok, str):
                        tokens.append(tok)
                    if isinstance(lp, (int, float)):
                        token_lps.append(float(lp))
                    top_by_pos.append(_normalize_topk(getattr(pos, "top_logprobs", None)))
                logprobs_obj = {
                    "tokens": tokens,
                    "token_logprobs": token_lps,
                    "top_logprobs": top_by_pos,
                }

        return _maybe_logprobs_return(text, "deepseek", model, logprobs_obj, want_k, finish)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        from openai import OpenAI
        api = os.getenv("DEEPSEEK_API_KEY")
        if not api:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        client = OpenAI(api_key=api, base_url="https://api.deepseek.com")
        return sorted([m.id for m in client.models.list().data])


@register_provider("anthropic")
class AnthropicProvider:
    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        import anthropic
        api = os.getenv("ANTHROPIC_API_KEY")
        if not api:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api)

        clean = [m for m in messages if (m.get("role") in ("user", "assistant"))]
        params = {
            "model": model,
            "system": system or None,
            "max_tokens": 512,
            "messages": clean,
        }
        # YENİ
        params["temperature"] = float(0.0 if temperature is None else temperature)


        resp = client.messages.create(**params)
        text = "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"]).strip()
        return _maybe_logprobs_return(text, "anthropic", model, None, int(want_logprobs) or 0)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        import anthropic
        api = os.getenv("ANTHROPIC_API_KEY")
        if not api:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        try:
            res = anthropic.Anthropic(api_key=api).models.list()
            ids = [getattr(m, "id", None) or getattr(m, "name", None) for m in getattr(res, "data", [])]
            return sorted({mid for mid in ids if mid})
        except Exception:
            return ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"]

@register_provider("google")
class GoogleProvider:
    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        import google.generativeai as genai
        api = os.getenv("GOOGLE_API_KEY")
        if not api:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api)

        parts = []
        if system:
            parts.append(system.strip())
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                parts.append(f"ASSISTANT: {content}")
            else:
                parts.append(f"USER: {content}")
        prompt = "\n\n".join(parts).strip()


        gen_cfg = {"max_output_tokens": 512}
        gen_cfg["temperature"] = float(0.0 if temperature is None else temperature)
        gen_cfg["top_p"] = float(1.0 if top_p is None else top_p)


        gm = genai.GenerativeModel(model_name=model, generation_config=gen_cfg)
        resp = gm.generate_content(prompt, request_options={"timeout": (timeout or 30)})

        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            out_text = text.strip()
        else:
            cands = getattr(resp, "candidates", None) or []
            out_text = ""
            if cands:
                content = getattr(cands[0], "content", None)
                prts = getattr(content, "parts", None) or []
                for p in prts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t.strip():
                        out_text = t.strip()
                        break
        return _maybe_logprobs_return(out_text, "google", model, None, int(want_logprobs) or 0)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        import google.generativeai as genai
        api = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api)
        models = genai.list_models()
        names = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                names.append((getattr(m, "name", None) or getattr(m, "model", None) or "").split("/")[-1])
        return sorted({n for n in names if n})

from typing import Any, Dict, Optional, Union

@register_provider("openrouter")
class OpenRouterProvider:
    """OpenRouter API provider - Access to 200+ models including Grok, Claude, GPT-4, Gemini."""
    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        from openai import OpenAI

        api = os.getenv("OPENROUTER_API_KEY")
        if not api:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(
            api_key=api,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout or 30,
        )

        # Check if this is an Anthropic model - they don't accept system role
        is_anthropic = "anthropic" in model.lower() or "claude" in model.lower()

        final_msgs = []
        if is_anthropic:
            # Anthropic models via OpenRouter only accept 'user' and 'assistant' roles
            # Prepend system message to first user message if present
            if system and messages:
                first_msg = messages[0]
                if first_msg.get("role") == "user":
                    # Prepend system to first user message
                    content = first_msg.get("content", "")
                    final_msgs.append({
                        "role": "user",
                        "content": f"{system}\n\n{content}"
                    })
                    final_msgs.extend(messages[1:])
                else:
                    # If first message isn't user, add system as user message
                    final_msgs.append({"role": "user", "content": system})
                    final_msgs.extend(messages)
            else:
                final_msgs.extend(messages)
        else:
            # Non-Anthropic models support system role
            if system:
                final_msgs.append({"role": "system", "content": system})
            final_msgs.extend(messages)

        params: Dict[str, Any] = {"model": model, "messages": final_msgs}

        # Anthropic models don't allow both temperature and top_p
        if is_anthropic:
            # Prefer temperature over top_p for Anthropic models
            params["temperature"] = float(0.0 if temperature is None else temperature)
        else:
            params["temperature"] = float(0.0 if temperature is None else temperature)
            params["top_p"] = float(1.0 if top_p is None else top_p)

        want_k = 5 if want_logprobs is True else (int(want_logprobs) if isinstance(want_logprobs, int) and want_logprobs > 0 else 0)
        if want_k:
            # OpenAI/OpenRouter API: logprobs is boolean, top_logprobs is the count
            params["logprobs"] = True
            params["top_logprobs"] = want_k

        try:
            resp = client.chat.completions.create(**params)
        except Exception as e:
            error_str = str(e).lower()
            if ("not allowed to request logprobs" in error_str or
                "logprobs" in error_str or
                "does not support" in error_str or
                "unsupported" in error_str):
                params.pop("logprobs", None)
                params.pop("top_logprobs", None)
                want_k = 0
                resp = client.chat.completions.create(**params)
            else:
                raise

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)

        logprobs_obj = None
        if want_k:
            log = getattr(choice, "logprobs", None)
            tokens, token_lps, top_by_pos = [], [], []
            if log:
                # Try different possible locations for logprobs content
                content = getattr(log, "content", None)
                if content is None:
                    # Some providers might put logprobs directly in the choice
                    content = getattr(choice, "logprobs_content", None)
                if content is None:
                    # Try alternative attribute names - some providers flatten the structure
                    content = getattr(log, "tokens", None)
                    if content:
                        # If we have tokens directly, reconstruct the structure
                        token_lps = getattr(log, "token_logprobs", [])
                        top_by_pos = getattr(log, "top_logprobs", [])
                        tokens = content
                        logprobs_obj = {
                            "tokens": tokens,
                            "token_logprobs": token_lps,
                            "top_logprobs": top_by_pos,
                        }
                    else:
                        content = None

                if content and not logprobs_obj:
                    for pos in content:
                        tok = getattr(pos, "token", None)
                        lp = getattr(pos, "logprob", None)
                        if isinstance(tok, str):
                            tokens.append(tok)
                        if isinstance(lp, (int, float)):
                            token_lps.append(float(lp))
                        top_by_pos.append(_normalize_topk(getattr(pos, "top_logprobs", None)))
                    logprobs_obj = {
                        "tokens": tokens,
                        "token_logprobs": token_lps,
                        "top_logprobs": top_by_pos,
                    }

        return _maybe_logprobs_return(text, "openrouter", model, logprobs_obj, want_k, finish)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        from openai import OpenAI
        api = os.getenv("OPENROUTER_API_KEY")
        if not api:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        
        # Initialize client
        client = OpenAI(
            api_key=api,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout or 30
        )
        
        try:
            return sorted([m.id for m in client.models.list().data])
        except Exception:
            # Return known popular models if API call fails
            return [
                "x-ai/grok-beta",
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-opus",
                "openai/gpt-4-turbo",
                "openai/gpt-4",
                "openai/gpt-3.5-turbo",
                "google/gemini-pro",
                "meta-llama/llama-3.1-405b-instruct",
                "deepseek/deepseek-chat",
            ]

@register_provider("aimlapi")
class AIMLAPIProvider:
    """AI/ML API provider - OpenAI-compatible API with access to 200+ models.

    Website: https://aimlapi.com
    Supports: Various models with logprobs capability
    API Format: OpenAI-compatible (uses OpenAI Python SDK)
    """
    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        from openai import OpenAI

        api = os.getenv("AIMLAPI_API_KEY")
        if not api:
            raise RuntimeError("AIMLAPI_API_KEY not set")

        # Initialize OpenAI client with AI/ML API base URL
        client = OpenAI(
            api_key=api,
            base_url="https://api.aimlapi.com/v1",
            timeout=timeout or 30,
        )

        # Check if this is an Anthropic model - they don't accept system role
        is_anthropic = "anthropic" in model.lower() or "claude" in model.lower()

        final_msgs = []
        if is_anthropic:
            # Anthropic models via AIMLAPI only accept 'user' and 'assistant' roles
            # Prepend system message to first user message if present
            if system and messages:
                first_msg = messages[0]
                if first_msg.get("role") == "user":
                    # Prepend system to first user message
                    content = first_msg.get("content", "")
                    final_msgs.append({
                        "role": "user",
                        "content": f"{system}\n\n{content}"
                    })
                    final_msgs.extend(messages[1:])
                else:
                    # If first message isn't user, add system as user message
                    final_msgs.append({"role": "user", "content": system})
                    final_msgs.extend(messages)
            else:
                final_msgs.extend(messages)
        else:
            # Non-Anthropic models support system role
            if system:
                final_msgs.append({"role": "system", "content": system})
            final_msgs.extend(messages)

        params: Dict[str, Any] = {"model": model, "messages": final_msgs}

        # Anthropic models don't allow both temperature and top_p
        if is_anthropic:
            # Prefer temperature over top_p for Anthropic models
            params["temperature"] = float(0.0 if temperature is None else temperature)
        else:
            params["temperature"] = float(0.0 if temperature is None else temperature)
            params["top_p"] = float(1.0 if top_p is None else top_p)

        want_k = 5 if want_logprobs is True else (int(want_logprobs) if isinstance(want_logprobs, int) and want_logprobs > 0 else 0)
        if want_k:
            # OpenAI-compatible API: logprobs is boolean, top_logprobs is the count
            params["logprobs"] = True
            params["top_logprobs"] = want_k

        try:
            resp = client.chat.completions.create(**params)
        except Exception as e:
            error_str = str(e).lower()
            if ("not allowed to request logprobs" in error_str or
                "logprobs" in error_str or
                "does not support" in error_str or
                "unsupported" in error_str):
                params.pop("logprobs", None)
                params.pop("top_logprobs", None)
                want_k = 0
                resp = client.chat.completions.create(**params)
            else:
                raise

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)

        logprobs_obj = None
        if want_k:
            log = getattr(choice, "logprobs", None)
            tokens, token_lps, top_by_pos = [], [], []
            if log:
                # Try different possible locations for logprobs content
                content = getattr(log, "content", None)
                if content is None:
                    # Some providers might put logprobs directly in the choice
                    content = getattr(choice, "logprobs_content", None)
                if content is None:
                    # Try alternative attribute names - some providers flatten the structure
                    content = getattr(log, "tokens", None)
                    if content:
                        # If we have tokens directly, reconstruct the structure
                        token_lps = getattr(log, "token_logprobs", [])
                        top_by_pos = getattr(log, "top_logprobs", [])
                        tokens = content
                        logprobs_obj = {
                            "tokens": tokens,
                            "token_logprobs": token_lps,
                            "top_logprobs": top_by_pos,
                        }
                    else:
                        content = None

                if content and not logprobs_obj:
                    for pos in content:
                        tok = getattr(pos, "token", None)
                        lp = getattr(pos, "logprob", None)
                        if isinstance(tok, str):
                            tokens.append(tok)
                        if isinstance(lp, (int, float)):
                            token_lps.append(float(lp))
                        top_by_pos.append(_normalize_topk(getattr(pos, "top_logprobs", None)))
                    logprobs_obj = {
                        "tokens": tokens,
                        "token_logprobs": token_lps,
                        "top_logprobs": top_by_pos,
                    }

        return _maybe_logprobs_return(text, "aimlapi", model, logprobs_obj, want_k, finish)

    def list_models(self, timeout: Optional[int]) -> List[str]:
        from openai import OpenAI
        api = os.getenv("AIMLAPI_API_KEY")
        if not api:
            raise RuntimeError("AIMLAPI_API_KEY not set")
        
        # Initialize client
        client = OpenAI(
            api_key=api,
            base_url="https://api.aimlapi.com/v1",
            timeout=timeout or 30
        )
        
        try:
            return sorted([m.id for m in client.models.list().data])
        except Exception:
            # Return empty list if API call fails
            return []

@register_provider("huggingface")
class HuggingFaceProvider:
    """Hugging Face Transformers provider for local/OS models with chat templates and optional logprobs."""
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}

    def _login_hf_if_needed(self) -> Optional[str]:
        from os import getenv
        token = getenv("HF_TOKEN") or getenv("HUGGINGFACEHUB_API_TOKEN") or getenv("HUGGINGFACE_TOKEN")
        if token:
            try:
                from huggingface_hub import login
                login(token=token)
            except Exception:
                pass
        return token

    def call(
        self, *, model, system, messages,
        temperature: Optional[float], top_p: Optional[float], timeout: Optional[int],
        want_logprobs: Union[bool, int] = False,
    ) -> Union[str, Dict[str, Any]]:
        import gc
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        token = self._login_hf_if_needed()

        # --- load (minimal & clean) ---
        if model not in self._model_cache:
            # tokenizer
            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, token=token, use_fast=True)
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            self._tokenizer_cache[model] = tok

            # dtype (no quant/env logic)
            torch_dtype = (
                torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
                else (torch.float16 if device == "cuda" else torch.float32)
            )

            from_pretrained_kwargs = dict(trust_remote_code=True, token=token)
            if device == "cuda":
                from_pretrained_kwargs["device_map"] = "auto"
            else:
                from_pretrained_kwargs["low_cpu_mem_usage"] = True
            from_pretrained_kwargs["torch_dtype"] = torch_dtype

            # pre-load cleanup (this fixed your OOM case)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            mdl = AutoModelForCausalLM.from_pretrained(model, **from_pretrained_kwargs)
            if device == "cpu":
                mdl = mdl.to(device)
            mdl.eval()
            self._model_cache[model] = mdl

        tok = self._tokenizer_cache[model]
        mdl = self._model_cache[model]

        # messages -> prompt
        chat_msgs = []
        if system:
            chat_msgs.append({"role": "system", "content": system})
        for m in messages:
            r = m.get("role", "user")
            c = (m.get("content") or "").strip()
            if c:
                chat_msgs.append({"role": r, "content": c})

        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            prompt_text = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
        else:
            parts = [f"{mm['role'].upper()}: {mm['content']}" for mm in chat_msgs]
            parts.append("ASSISTANT:")
            prompt_text = "\n\n".join(parts)

        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        inputs = tok(prompt_text, return_tensors="pt", padding=True).to(device)

        # generation (no env switches)
        k = 5 if want_logprobs is True else (int(want_logprobs) if isinstance(want_logprobs, int) and want_logprobs > 0 else 0)
        want_scores = bool(k)

        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": (temperature is not None and temperature > 0),
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "return_dict_in_generate": want_scores,
            "output_scores": want_scores,
            # keep this to lower VRAM pressure in long gens
            "use_cache": False,
        }
        if temperature is not None and temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            out = mdl.generate(**inputs, **gen_kwargs)

        if want_scores:
            seq = out.sequences[0]
            gen_ids = seq[inputs["input_ids"].shape[1]:]
        else:
            gen_ids = out[0][inputs["input_ids"].shape[1]:]

        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        logprobs_obj = None
        if want_scores:
            try:
                scores = out.scores  # list of [batch, vocab]
                logdists = [F.log_softmax(s, dim=-1) for s in scores]

                chosen_ids = gen_ids
                chosen_logprobs = []
                top_by_pos = []

                for i, tok_id in enumerate(chosen_ids):
                    lp = float(logdists[i][0, tok_id])
                    chosen_logprobs.append(lp)

                    if k <= 0:
                        top_by_pos.append([])
                        continue

                    vals, idxs = torch.topk(logdists[i][0], k)
                    cand_tokens = tok.convert_ids_to_tokens(idxs.tolist())
                    cand = [{"token": cand_tokens[j], "logprob": float(vals[j])} for j in range(len(cand_tokens))]
                    top_by_pos.append(cand)

                tokens_str = tok.convert_ids_to_tokens(chosen_ids.tolist())
                logprobs_obj = {
                    "tokens": tokens_str,
                    "token_logprobs": chosen_logprobs,
                    "top_logprobs": top_by_pos,
                }
            except Exception:
                logprobs_obj = None

        return _maybe_logprobs_return(text, "huggingface", model, logprobs_obj, k)



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    cl = LLMClient("openai", "gpt-5", temperature=1.0, top_p=1.0)
    res = cl.ask("just say 'hi'", want_logprobs=0)
    print(res)
    for i in cl.list_models():
        print(i)
