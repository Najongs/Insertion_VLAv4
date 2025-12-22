#!/usr/bin/env python
"""
Fix config.json format for lerobot compatibility
Changes 'model_type' to 'type' key
"""

import json
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
import tempfile
import shutil

def fix_config(repo_id: str, token: str = None):
    """
    Download config.json, fix the format, and re-upload

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token (needed for private repos)
    """
    print(f"Downloading config.json from {repo_id}...")

    # Download current config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        token=token,
    )

    # Load and fix config
    with open(config_path) as f:
        config = json.load(f)

    print("\nOriginal config:")
    print(json.dumps(config, indent=2))

    # Fix the config
    if "model_type" in config and "type" not in config:
        config["type"] = config["model_type"]
        print("\n✓ Added 'type' key from 'model_type'")
    elif "type" not in config:
        config["type"] = "smolvla"
        print("\n✓ Added 'type': 'smolvla'")
    else:
        print("\n✓ Config already has 'type' key")

    print("\nFixed config:")
    print(json.dumps(config, indent=2))

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        temp_path = f.name

    # Upload fixed config
    print(f"\nUploading fixed config to {repo_id}...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo="config.json",
        repo_id=repo_id,
        token=token,
        commit_message="Fix config.json format for lerobot compatibility (type key)",
    )

    # Cleanup
    Path(temp_path).unlink()

    print("\n✓ Config fixed and uploaded successfully!")
    print("\nYou can now run download_model.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix config.json format")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")

    args = parser.parse_args()

    fix_config(args.repo_id, args.token)
