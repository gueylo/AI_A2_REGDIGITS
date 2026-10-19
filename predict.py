#!/usr/bin/env python3
"""Run one validated prediction from a JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inference import ModelService
from src.validation import FarmInput


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path, dest="json_path")
    args = parser.parse_args()
    try:
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
        farm = FarmInput.model_validate(payload)
        result = ModelService(ROOT).predict(farm)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

