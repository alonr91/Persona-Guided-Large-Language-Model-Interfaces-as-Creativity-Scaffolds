"""HF transformers LLM client with Pydantic-schema-constrained JSON decoding.

Uses lm-format-enforcer to constrain the token distribution during generation
so that the model's output is guaranteed to match the given Pydantic schema.
"""
from __future__ import annotations
import json, torch
from typing import Type, TypeVar
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# optimum-intel is optional; only imported if we use the OpenVINO backend.
try:
    from optimum.intel import OVModelForCausalLM  # type: ignore
    _HAS_OV = True
except Exception:
    _HAS_OV = False

# --- compat shim: lm-format-enforcer expects PreTrainedTokenizerBase at the
# old import path (transformers<5). Transformers 5.x moved it.
import transformers.tokenization_utils as _tok_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
if not hasattr(_tok_utils, 'PreTrainedTokenizerBase'):
    _tok_utils.PreTrainedTokenizerBase = _PTB

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from . import config

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """Singleton-ish wrapper around a local HF causal LM."""

    _tokenizer = None
    _model = None

    @classmethod
    def load(cls) -> None:
        if cls._model is not None:
            return
        path = str(config.MODEL_DIR)
        backend = getattr(config, 'LLM_BACKEND', 'transformers-cpu')
        print(f'[llm_client] loading {path} via backend={backend}')
        cls._tokenizer = AutoTokenizer.from_pretrained(path)

        if backend == 'openvino':
            if not _HAS_OV:
                raise RuntimeError('optimum-intel not installed; cannot use openvino backend')
            device = getattr(config, 'OV_DEVICE', 'GPU')
            print(f'[llm_client] OpenVINO device={device}')
            cls._model = OVModelForCausalLM.from_pretrained(path, device=device)
        else:
            # Gemma 4 multimodal support (CPU fallback only)
            cfg = AutoConfig.from_pretrained(path)
            if getattr(cfg, 'model_type', '') == 'gemma4':
                from transformers import Gemma4ForConditionalGeneration
                print('[llm_client] multimodal gemma4 detected; loading full checkpoint')
                cls._model = Gemma4ForConditionalGeneration.from_pretrained(
                    path, dtype=torch.bfloat16,
                )
            else:
                cls._model = AutoModelForCausalLM.from_pretrained(
                    path, dtype=torch.bfloat16,
                )
            cls._model.to('cpu').eval()
        print('[llm_client] ready.')

    @classmethod
    def generate_json(
        cls,
        schema_cls: Type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 320,
    ) -> tuple[T | None, dict]:
        """Generate schema-valid JSON. Returns (parsed_object | None, debug_info)."""
        cls.load()
        tok = cls._tokenizer
        model = cls._model

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors='pt')

        parser = JsonSchemaParser(schema_cls.model_json_schema())
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(tok, parser)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=tok.eos_token_id,
        )
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        new_tokens = out[0, inputs['input_ids'].shape[1]:]
        raw = tok.decode(new_tokens, skip_special_tokens=True).strip()

        debug = {'raw_output': raw, 'n_input_tokens': int(inputs['input_ids'].shape[1]),
                 'n_output_tokens': int(new_tokens.shape[0])}
        try:
            obj = schema_cls.model_validate_json(raw)
            debug['valid_json'] = True
            return obj, debug
        except Exception as e:
            # rare thanks to enforcer, but the parser can still emit truncated JSON if
            # max_new_tokens is hit
            debug['valid_json'] = False
            debug['parse_error'] = str(e)
            return None, debug
