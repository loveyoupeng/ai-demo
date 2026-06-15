#!/usr/bin/env python3
"""Download TinyStories dataset and save to resource/ directory.

Usage:
    uv run scripts/download_tinystories.py

Requires:
    datasets  (pip install datasets)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package required. Install with:\n  pip install datasets", file=sys.stderr)
    sys.exit(1)

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"


def main() -> None:
    """Download TinyStories and save as JSON files."""
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading TinyStories (train)...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_stories: list[str] = []
    for i, example in enumerate(train_ds):
        text = example.get("text", "")
        if text and text.strip():
            train_stories.append(text)
            if i >= 10000:
                break
    print(f"  Train: {len(train_stories)} stories")

    print("Downloading TinyStories (validation)...")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_stories: list[str] = []
    for i, example in enumerate(val_ds):
        text = example.get("text", "")
        if text and text.strip():
            val_stories.append(text)
            if i >= 50:
                break
    print(f"  Validation: {len(val_stories)} stories")

    train_file = RESOURCE_DIR / "tinystories_train.json"
    val_file = RESOURCE_DIR / "tinystories_val.json"

    with open(train_file, "w") as f:
        json.dump(train_stories, f)
    print(f"Saved {train_file} ({train_file.stat().st_size / 1024:.1f} KB)")

    with open(val_file, "w") as f:
        json.dump(val_stories, f)
    print(f"Saved {val_file} ({val_file.stat().st_size / 1024:.1f} KB)")

    print("Done.")


if __name__ == "__main__":
    main()
