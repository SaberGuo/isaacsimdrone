#!/usr/bin/env python3
"""Helper to check if an experiment with matching params already exists in logs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def extract_first_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    count = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            count += 1
        elif c == "}":
            count -= 1
        if count == 0:
            try:
                return json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--enable-apf", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--apf-attractive-weight", type=float, required=True)
    parser.add_argument("--apf-repulsive-weight", type=float, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    args = parser.parse_args()

    target = {
        "enable_apf": args.enable_apf,
        "apf_attractive_weight": args.apf_attractive_weight,
        "apf_repulsive_weight": args.apf_repulsive_weight,
        "timesteps": args.timesteps,
        "num_envs": args.num_envs,
    }

    # Default values for fields that may be missing in older runs
    defaults = {
        "enable_apf": False,
        "apf_attractive_weight": 0.5,
        "apf_repulsive_weight": -0.5,
    }

    for config_path in glob.glob(os.path.join(args.logs_dir, "*/config/config.txt")):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                text = f.read()
            data = extract_first_json(text)
            if data is None:
                continue
            match = all(data.get(k, defaults.get(k)) == v for k, v in target.items())
            if match:
                run_dir = os.path.dirname(os.path.dirname(config_path))
                if glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
                    print("FOUND")
                    return 0
        except Exception:
            continue

    print("NOT_FOUND")
    return 0


if __name__ == "__main__":
    sys.exit(main())
