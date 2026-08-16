"""Smoke-test configuration: paths, model, thresholds, schema."""
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
LOGS_CSV = ROOT / 'Experiment1_logs.csv'
MODELS_DIR = ROOT / 'models'
# Smoke round 3: OpenVINO INT4 on Intel Arc 140T GPU (~20 tok/s on 1.5B,
# est. ~8-12 tok/s on 4B-class). Upgraded from CPU bf16 (~0.5 tok/s).
MODEL_DIR = MODELS_DIR / 'qwen3-4b-instruct-2507-ov-int4'
LLM_BACKEND = 'openvino'        # 'openvino' | 'transformers-cpu'
OV_DEVICE = 'GPU'               # 'GPU' (Arc) | 'CPU' | 'NPU'
EMBED_MODEL_NAME = 'BAAI/bge-large-en-v1.5'        # auto-downloaded on first use
OUT_DIR = ROOT / 'analysis_out' / 'smoke'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR = OUT_DIR / 'smoke_transcripts'
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Agent 1 decoding
AGENT1_TEMPERATURE = 0.2
AGENT1_MAX_NEW_TOKENS = 480          # bumped from 320 after smoke-round-1 truncations
AGENT1_MAX_CONTEXT_CHARS = 2500      # crude cap on prompt size
AGENT1_MIN_EVIDENCE_WORDS = 3        # drop single-word / trivial evidence spans

# Agent 2 consolidation
CONSOLIDATION_SIM_THRESHOLD = 0.85  # merge candidates with cos sim >= this (raised from 0.80 to avoid over-merging distinct ideas like "cycling clothes" and "helmets")

# Agent 3 grounding
FUZZY_PARTIAL_RATIO_THRESHOLD = 90  # rapidfuzz partial_ratio cutoff

# Smoke-test selection
N_CONVERSATIONS_PER_CONDITION = 1   # one per (GPT, Divergent, Convergent, Rational, BoundedRational)
RANDOM_SEED = 7

# -------- Gemini (Scorer C, cross-model agreement) --------
# API key is read from environment (.env): GEMINI_API_KEY.
# Never hardcode. Model ID is parameterised so it can be swapped without code changes.
GEMINI_MODEL_ID = 'gemini-3.1-flash-lite-preview'
GEMINI_TEMPERATURE = 0.15
GEMINI_MAX_OUTPUT_TOKENS = 2200
GEMINI_REQUEST_TIMEOUT_S = 120
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BACKOFF_S = 4.0


# -------- Schemas --------
class IdeaCandidate(BaseModel):
    title: str = Field(description='short noun phrase, <=10 words')
    description: str = Field(description='1-2 sentences describing the user\'s proposal')
    evidence_span: str = Field(description='verbatim substring copied from the USER message')
    confidence: float = Field(description='extractor confidence as decimal 0.0-1.0')


class Extraction(BaseModel):
    ideas: list[IdeaCandidate] = Field(default_factory=list, description='0-3 candidate ideas, empty if user did not propose')


class CanonicalIdea(BaseModel):
    title: str
    description: str
    evidence_quotes: list[str] = Field(default_factory=list)
    source_candidate_ids: list[int] = Field(default_factory=list)


class Canonicalization(BaseModel):
    title: str
    description: str
