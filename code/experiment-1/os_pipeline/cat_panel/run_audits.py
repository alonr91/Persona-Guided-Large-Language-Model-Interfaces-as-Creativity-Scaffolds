"""Single-command audit runner — V4 (length-bias), V6 (halo), V7 (construct validity).

Run AFTER scoring + adjudication is complete. Mask-leak audit (V5)
issues additional Gemini calls and is run separately via
audit_mask_leak.py.
"""
from __future__ import annotations
import sys

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass


def main():
    print('=' * 60); print('V4 — length-bias regression'); print('=' * 60)
    from os_pipeline.cat_panel import audit_length_bias
    audit_length_bias.main()

    print('\n' + '=' * 60); print('V6 — halo-bias audit'); print('=' * 60)
    from os_pipeline.cat_panel import audit_halo
    audit_halo.main()

    print('\n' + '=' * 60); print('V7 — construct-validity audit'); print('=' * 60)
    from os_pipeline.cat_panel import construct_validity
    construct_validity.main()


if __name__ == '__main__':
    main()
