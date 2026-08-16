"""CAT-Panel-specific configuration.

Pinned independently from os_pipeline.config so changes here do NOT
disturb the existing §2.4 regulated rubric Scorer C (which still uses
the model in os_pipeline.config.GEMINI_MODEL_ID).
"""

# Single-model mode per user directive: use ONLY gemini-3.1-flash-lite-preview.
# (We previously had ['gemini-3.5-flash', 'gemini-3.1-flash-lite-preview']; the
# 3.5-flash slot is dropped because (a) the single-project key has very low
# 3.5-flash quota and (b) the user explicitly asked to continue on the 3.1
# model only. Keeping a single-model preference list also gives a cleaner
# methodology story: every row of the final corpus was scored on the same
# model.)
GEMINI_MODEL_PREFERENCES = ['gemini-3.1-flash-lite-preview']
GEMINI_MODEL_ID = GEMINI_MODEL_PREFERENCES[0]

# Conservative generation parameters: temperature 0.0 for reproducibility.
# We keep max_output_tokens generous (3000) since the new brevity rule in
# the judge prompts ensures responses stay short; the headroom is just to
# avoid mid-JSON truncation if the model occasionally exceeds the brevity
# instruction.
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_OUTPUT_TOKENS = 3000
