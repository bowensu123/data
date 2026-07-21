#!/usr/bin/env python3
"""Thin wrapper so `python run_pipeline.py ...` keeps working.

The implementation now lives in `aura_data_engine/cli.py`. After
`pip install -e .` you can also just run the `aura-pipeline` command.
"""

import sys

from aura_data_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())
