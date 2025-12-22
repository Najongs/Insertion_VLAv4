#!/usr/bin/env python
"""
Download model from Hugging Face and fix config.json locally
"""

import json
from huggingface_hub import snapshot_download
from pathlib import Path

def download_and_fix(repo_id: str, output_dir: str):
    """Download model and fix config.json locally"""

    print(f"Downloading model from: {repo_id}")
    print(f"Output directory: {output_dir}")

    # Download entire model
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )

    print(f"✓ Model downloaded to: {local_dir}")

    # Fix config.json
    config_path = Path(local_dir) / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        print("\nOriginal config:")
        print(json.dumps(config, indent=2))

        # Add 'type' key if missing
        if "type" not in config:
            if "model_type" in config:
                config["type"] = config["model_type"]
                print("\n✓ Added 'type' key from 'model_type'")
            else:
                config["type"] = "smolvla"
                print("\n✓ Added 'type': 'smolvla'")

        # Save fixed config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print("\n✓ Config.json fixed!")
        print("\nFixed config:")
        with open(config_path) as f:
            print(f.read())
    else:
        print("⚠ config.json not found")

    print(f"\n{'='*60}")
    print("SUCCESS! Model ready for inference")
    print(f"{'='*60}")
    print(f"Model location: {local_dir}")
    print(f"Use this path in your inference script")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="downloads/model")

    args = parser.parse_args()

    download_and_fix(args.repo_id, args.output_dir)
