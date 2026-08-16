"""Tiny smoke test for GeminiClient. Verifies API key, model name, and
schema-constrained JSON output before running the full 50-episode pass."""
from os_pipeline.gemini_client import GeminiClient
from os_pipeline.regulated.rubric import BundledEpisodeScore

GeminiClient.load()
obj, dbg = GeminiClient.generate_json(
    BundledEpisodeScore,
    'You output JSON only.',
    'Return a BundledEpisodeScore with conversation_id=1, '
    'episode_id="test", and an empty scores list.',
    max_new_tokens=400,
)
print('valid_json :', dbg.get('valid_json'))
print('parse_path :', dbg.get('parse_path'))
print('parse_err  :', dbg.get('parse_error'))
print('raw[:400]  :', (dbg.get('raw_output') or '')[:400])
print('parsed     :', obj)
