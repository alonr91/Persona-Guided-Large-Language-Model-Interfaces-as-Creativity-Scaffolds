"""Gemini client with Pydantic-schema-constrained JSON output.

Supports a **pool of API keys** with automatic rotation on 429
RESOURCE_EXHAUSTED responses. Keys are discovered from environment
variables named GEMINI_API_KEY, GEMINI_API_KEY_2, ..., GEMINI_API_KEY_8.
On Windows, if the current shell did not inherit these vars (because
they were added after the shell started), the loader falls back to a
fresh registry read via PowerShell.

When a key returns 429 the client marks it as "cooling" for
KEY_COOLDOWN_S seconds and immediately rotates to the next available
key. If all keys are cooling, the client sleeps until the earliest
cooldown expires and tries again. This lets a long-running scoring job
keep going across the daily-quota boundary by hot-swapping keys.

Used as Scorer C in §2.4 (single-key path is preserved for backward
compatibility) and as the engine of the CAT-Panel layer (multi-key
path).
"""
from __future__ import annotations
import os
import time
import json
import sys
import subprocess
import threading
from typing import Type, TypeVar
from pydantic import BaseModel

# Optional .env loading; harmless if python-dotenv is not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# google-genai is the current Gemini SDK. Import lazily.
_genai = None
_genai_types = None

from . import config

T = TypeVar('T', bound=BaseModel)


def _lazy_import_genai():
    global _genai, _genai_types
    if _genai is not None:
        return
    try:
        from google import genai as _g
        from google.genai import types as _gt
    except ImportError as e:
        raise ImportError(
            "google-genai is not installed. Run: pip install google-genai python-dotenv"
        ) from e
    _genai = _g
    _genai_types = _gt


# ----------------------------------------------------------------------
# Key discovery
# ----------------------------------------------------------------------

# COMPLIANCE NOTE (per the Google APIs Terms of Service, Section 2(d)):
# We use a single API key from a single Google Cloud project to respect
# per-project quota. The numbered keys (GEMINI_API_KEY_2 ... _15) are
# NOT discovered by this client. Restoring the multi-key code requires
# explicit re-enabling and only makes sense after the suspended projects
# are reinstated and the project consolidation in the appeal is in place.
KEY_ENV_NAMES = ('GEMINI_API_KEY',)


_winreg_cache: dict[str, str] | None = None


def _enumerate_windows_user_env() -> dict[str, str]:
    """One-shot read of every GEMINI_API_KEY* user env var from the
    Windows registry. Cached after first call.

    Reading the registry directly is dramatically faster than 50
    separate PowerShell invocations at startup. The User-scope
    Environment hive lives at HKEY_CURRENT_USER\\Environment.
    """
    global _winreg_cache
    if _winreg_cache is not None:
        return _winreg_cache
    out: dict[str, str] = {}
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                i = 0
                while True:
                    try:
                        name, value, _ = winreg.EnumValue(k, i)
                    except OSError:
                        break
                    if name and name.upper().startswith('GEMINI_API_KEY') and value:
                        # Normalise to canonical upper-case name so case
                        # typos in registry entries (e.g. "GEMINI_API_key_2")
                        # are still discoverable via the canonical lookup.
                        out[name.upper()] = str(value)
                    i += 1
        except Exception:
            pass
    _winreg_cache = out
    return out


def _discover_keys() -> list[tuple[str, str]]:
    """Return [(env_var_name, key_value), ...] for every populated GEMINI key.

    Tries os.environ first (cheapest), then the cached Windows registry
    enumeration as fallback. This way, keys added to Windows after the
    current shell started are still discovered on each fresh process.
    """
    reg = _enumerate_windows_user_env()
    found: list[tuple[str, str]] = []
    for name in KEY_ENV_NAMES:
        v = os.environ.get(name)
        if not v:
            v = reg.get(name)
        if v:
            found.append((name, v))
    return found


# ----------------------------------------------------------------------
# Key-pool with cooldown
# ----------------------------------------------------------------------

# Cooldown when a key returns 429. Most 429s on Gemini free tier are RPM
# violations that reset within ~60 seconds; only the daily-cap 429 is
# longer-lived. We use a short cooldown (90s) so RPM-cooled keys are
# tried again quickly; a key that keeps 429ing will just re-cool, which
# is the correct behaviour for a daily-cap-exhausted key.
KEY_COOLDOWN_S = 90


class _KeyPool:
    """Per-(key, model) cooldown pool with thread-safe acquisition.

    When the scorer wants to make a call it provides an ordered list of
    model preferences (e.g. ['gemini-3.5-flash',
    'gemini-3.1-flash-lite-preview']). The pool finds the BEST available
    (key, model) combination — primary model first, falling back through
    the preference list. Each (key, model) combination has its own
    cooldown clock, so a key that's daily-exhausted on gemini-3.5-flash
    can still serve gemini-3.1-flash-lite-preview from the same pool.
    """

    def __init__(self, keys: list[tuple[str, str]]):
        if not keys:
            raise RuntimeError(
                "No GEMINI_API_KEY* env vars found. Set at least one of "
                f"{KEY_ENV_NAMES}, then restart."
            )
        self.keys = keys              # [(name, value), ...]
        self.idx = 0
        self.clients: list = [None] * len(keys)
        self._lock = threading.Lock()
        # Per-key, per-model cooldown — populated lazily as new models appear.
        # _cooldown[(key_idx, model_name)] -> epoch seconds until ready
        self._cooldown: dict[tuple[int, str], float] = {}
        # Per-key in-flight counter (independent of model)
        self.in_flight: list[int] = [0] * len(keys)
        self.last_used_at: list[float] = [0.0] * len(keys)

    def _now(self) -> float:
        return time.time()

    def _client_for(self, i: int):
        if self.clients[i] is None:
            self.clients[i] = _genai.Client(api_key=self.keys[i][1])
        return self.clients[i]

    def _is_cooling(self, key_idx: int, model: str, now: float) -> bool:
        return self._cooldown.get((key_idx, model), 0.0) > now

    def acquire(self, model_preferences: list[str]) -> tuple[int, str, str, object]:
        """Return (key_idx, key_name, model, client) for the best available
        (key, model) combination. Tries each model in preference order;
        within a model, picks the key with the lowest in-flight count."""
        if not model_preferences:
            raise ValueError('model_preferences must be a non-empty list')
        while True:
            with self._lock:
                now = self._now()
                for model in model_preferences:
                    eligible = [i for i in range(len(self.keys))
                                if not self._is_cooling(i, model, now)]
                    if eligible:
                        best = min(eligible,
                                   key=lambda i: (self.in_flight[i],
                                                  self.last_used_at[i]))
                        self.in_flight[best] += 1
                        self.last_used_at[best] = now
                        return best, self.keys[best][0], model, self._client_for(best)
                # all (key, model) combos cooling — find the soonest unlock
                soonest = min(
                    (self._cooldown.get((i, m), 0.0)
                     for i in range(len(self.keys)) for m in model_preferences),
                    default=now + 60.0,
                )
            wait = max(1.0, soonest - self._now())
            wait = min(wait, 60.0)
            print(f'[gemini_client] all {len(self.keys)} keys cooling on '
                  f'{model_preferences}; sleeping {wait:.0f}s...')
            time.sleep(wait)

    def release(self, key_idx: int) -> None:
        with self._lock:
            self.in_flight[key_idx] = max(0, self.in_flight[key_idx] - 1)

    def mark_quota_exhausted(self, key_idx: int, model: str, reason: str = '') -> None:
        with self._lock:
            self._cooldown[(key_idx, model)] = self._now() + KEY_COOLDOWN_S
        print(f'[gemini_client] key #{key_idx+1} ({self.keys[key_idx][0]}) '
              f'× {model} → cooling for {KEY_COOLDOWN_S}s  ({reason[:80]})')

    def summary(self, model_preferences: list[str] | None = None) -> str:
        now = self._now()
        if model_preferences is None:
            # any-model readiness — count keys with at least one model free
            ready = sum(1 for i in range(len(self.keys))
                        if any(self._cooldown.get((i, m), 0.0) <= now
                               for m in {k[1] for k in self._cooldown}) or
                        not any(self._cooldown.get((i, m), 0.0) > now
                                for m in {k[1] for k in self._cooldown}))
            return f'{ready}/{len(self.keys)} keys ready (any-model)'
        per_model = []
        for m in model_preferences:
            n_ready = sum(1 for i in range(len(self.keys))
                          if not self._is_cooling(i, m, now))
            per_model.append(f'{m.split("-")[-1]}={n_ready}')
        return f'{len(self.keys)} keys; ready ' + ' / '.join(per_model)


# ----------------------------------------------------------------------
# Client (singleton wrapper around the pool)
# ----------------------------------------------------------------------

class GeminiClient:
    """Wrapper around Google Gemini with multi-key rotation."""

    _pool: _KeyPool | None = None
    _load_lock = threading.Lock()

    @classmethod
    def load(cls) -> None:
        if cls._pool is not None:
            return
        with cls._load_lock:
            if cls._pool is not None:  # double-checked locking
                return
            _lazy_import_genai()
            keys = _discover_keys()
            if not keys:
                raise RuntimeError(
                    "No GEMINI_API_KEY* env vars found. Add GEMINI_API_KEY (and "
                    "optionally GEMINI_API_KEY_2..8) to your environment."
                )
            cls._pool = _KeyPool(keys)
            names = ', '.join(n for n, _ in keys)
            print(f'[gemini_client] ready: {len(keys)} keys loaded ({names}); '
                  f'model={config.GEMINI_MODEL_ID}')

    @classmethod
    def pool_summary(cls, model_preferences: list[str] | None = None) -> str:
        return cls._pool.summary(model_preferences) if cls._pool else '(unloaded)'

    @classmethod
    def generate_json(
        cls,
        schema_cls: Type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        model_id: str | None = None,
        model_preferences: list[str] | None = None,
    ) -> tuple[T | None, dict]:
        """Generate schema-valid JSON via Gemini.

        Returns (parsed_object | None, debug_info).
        `debug_info['model_used']` reports which model in `model_preferences`
        actually produced the response.

        Behaviour on errors:
          - 429 RESOURCE_EXHAUSTED → mark (key, model) as cooling and try
            another (key, model) combination from `model_preferences`.
          - Other transient errors (5xx, timeouts) → retry with exponential
            backoff on the SAME (key, model) pair up to GEMINI_MAX_RETRIES.
          - Schema validation errors → no retry, return None with parse_error.
        """
        cls.load()
        _lazy_import_genai()
        pool = cls._pool

        temp = config.GEMINI_TEMPERATURE if temperature is None else temperature
        max_tok = config.GEMINI_MAX_OUTPUT_TOKENS if max_new_tokens is None else max_new_tokens

        # Build the model-preference list. If the caller passed a single
        # model_id, that becomes the sole preference (backward compatible
        # with Scorer C). Otherwise we use the caller-supplied preferences,
        # falling back to global config.GEMINI_MODEL_ID.
        if model_preferences:
            preferences = list(model_preferences)
        elif model_id is not None:
            preferences = [model_id]
        else:
            preferences = [config.GEMINI_MODEL_ID]

        gen_config = _genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=float(temp),
            max_output_tokens=int(max_tok),
            response_mime_type='application/json',
            response_schema=schema_cls,
        )

        raw = ''
        last_err: Exception | None = None
        attempt = 0
        key_attempts_used = 0
        # Try every (key, model) combination at most once per outer call
        MAX_KEY_ROTATIONS = len(pool.keys) * len(preferences)
        active_model: str = preferences[0]
        resp = None

        while attempt < config.GEMINI_MAX_RETRIES:
            i, key_name, active_model, client = pool.acquire(preferences)
            try:
                resp = client.models.generate_content(
                    model=active_model,
                    contents=user_prompt,
                    config=gen_config,
                )
                raw = (resp.text or '').strip()
                pool.release(i)
                break
            except Exception as e:
                pool.release(i)
                last_err = e
                msg = str(e).lower()
                is_quota = ('429' in msg or 'resource_exhausted' in msg
                            or 'quota' in msg)
                is_transient = is_quota or any(s in msg for s in (
                    'timeout', 'deadline', '500', '502', '503', '504',
                    'unavailable',
                ))
                if is_quota:
                    pool.mark_quota_exhausted(i, active_model, str(e)[:80])
                    key_attempts_used += 1
                    if key_attempts_used >= MAX_KEY_ROTATIONS:
                        debug = {
                            'raw_output': '', 'valid_json': False,
                            'parse_error': f'AllKeysExhausted: {type(e).__name__}: {e}',
                            'n_attempts': attempt + 1,
                            'key_attempts': key_attempts_used,
                            'model_used': active_model,
                        }
                        return None, debug
                    continue
                if is_transient and attempt < config.GEMINI_MAX_RETRIES - 1:
                    backoff = config.GEMINI_RETRY_BACKOFF_S * (2 ** attempt)
                    time.sleep(backoff)
                    attempt += 1
                    continue
                debug = {
                    'raw_output': '', 'valid_json': False,
                    'parse_error': f'{type(e).__name__}: {e}',
                    'n_attempts': attempt + 1,
                    'key_attempts': key_attempts_used,
                    'model_used': active_model,
                }
                return None, debug

        debug = {'raw_output': raw, 'n_attempts': attempt + 1,
                 'key_attempts': key_attempts_used,
                 'model_used': active_model}

        # Gemini may return parsed object directly via .parsed when schema is set.
        try:
            parsed = getattr(resp, 'parsed', None)
            if parsed is not None and isinstance(parsed, schema_cls):
                debug['valid_json'] = True
                debug['parse_path'] = 'sdk_parsed'
                return parsed, debug
        except Exception:
            pass

        # Fallback: parse the raw text ourselves.
        try:
            obj = schema_cls.model_validate_json(raw)
            debug['valid_json'] = True
            debug['parse_path'] = 'manual_validate'
            return obj, debug
        except Exception as e:
            stripped = raw
            if stripped.startswith('```'):
                stripped = stripped.split('\n', 1)[-1]
                if stripped.endswith('```'):
                    stripped = stripped.rsplit('```', 1)[0]
                stripped = stripped.strip()
            try:
                obj = schema_cls.model_validate_json(stripped)
                debug['valid_json'] = True
                debug['parse_path'] = 'fence_stripped'
                return obj, debug
            except Exception as e2:
                debug['valid_json'] = False
                debug['parse_error'] = f'{type(e2).__name__}: {e2}'
                return None, debug
