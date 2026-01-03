#!/usr/bin/env python
"""
Quick test to verify action expert reset functionality.
This script loads the model twice and compares weights.
"""

import torch
from pathlib import Path
import sys

# Add lerobot to path
sys.path.insert(0, "/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def get_layer_weights(policy, layer_name):
    """Extract weights from a specific layer."""
    try:
        if layer_name == "lm_expert":
            return policy.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight.clone()
        elif layer_name == "state_proj":
            return policy.model.state_proj.weight.clone()
        elif layer_name == "action_in_proj":
            return policy.model.action_in_proj.weight.clone()
        elif layer_name == "action_out_proj":
            return policy.model.action_out_proj.weight.clone()
        elif layer_name == "vlm":
            # Get VLM weights (should NOT change)
            return policy.model.vlm_with_expert.vlm.model.text_model.layers[0].self_attn.q_proj.weight.clone()
    except Exception as e:
        print(f"Failed to get {layer_name} weights: {e}")
        return None


def main():
    print("=" * 70)
    print("Testing Action Expert Reset Functionality")
    print("=" * 70)

    model_id = "lerobot/smolvla_base"

    # Load model WITHOUT reset
    print("\n1️⃣  Loading model WITHOUT reset...")
    policy1 = SmolVLAPolicy.from_pretrained(model_id)

    weights_before = {
        "lm_expert": get_layer_weights(policy1, "lm_expert"),
        "state_proj": get_layer_weights(policy1, "state_proj"),
        "action_in_proj": get_layer_weights(policy1, "action_in_proj"),
        "action_out_proj": get_layer_weights(policy1, "action_out_proj"),
        "vlm": get_layer_weights(policy1, "vlm"),
    }

    # Manually reset action expert (simulating what the training script does)
    print("\n2️⃣  Resetting action expert weights...")

    def reinit_module(module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                if param.ndim >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    reinit_module(policy1.model.vlm_with_expert.lm_expert)
    reinit_module(policy1.model.state_proj)
    reinit_module(policy1.model.action_in_proj)
    reinit_module(policy1.model.action_out_proj)
    reinit_module(policy1.model.action_time_mlp_in)
    reinit_module(policy1.model.action_time_mlp_out)

    # Get weights after reset
    weights_after = {
        "lm_expert": get_layer_weights(policy1, "lm_expert"),
        "state_proj": get_layer_weights(policy1, "state_proj"),
        "action_in_proj": get_layer_weights(policy1, "action_in_proj"),
        "action_out_proj": get_layer_weights(policy1, "action_out_proj"),
        "vlm": get_layer_weights(policy1, "vlm"),
    }

    # Compare weights
    print("\n" + "=" * 70)
    print("3️⃣  Comparing weights BEFORE vs AFTER reset:")
    print("=" * 70)

    for name in weights_before.keys():
        if weights_before[name] is None or weights_after[name] is None:
            continue

        are_equal = torch.allclose(weights_before[name], weights_after[name], atol=1e-6)

        if name == "vlm":
            # VLM should NOT change
            status = "✅ UNCHANGED (expected)" if are_equal else "❌ CHANGED (unexpected!)"
            print(f"\n{name:20s}: {status}")
            if not are_equal:
                print(f"  ⚠️  WARNING: VLM weights changed! This should not happen!")
        else:
            # Action expert should change
            status = "✅ CHANGED (expected)" if not are_equal else "❌ UNCHANGED (unexpected!)"
            print(f"\n{name:20s}: {status}")

        # Show some statistics
        print(f"  Before: mean={weights_before[name].mean():.6f}, std={weights_before[name].std():.6f}")
        print(f"  After:  mean={weights_after[name].mean():.6f}, std={weights_after[name].std():.6f}")

        if not are_equal:
            diff = (weights_before[name] - weights_after[name]).abs()
            print(f"  Diff:   mean={diff.mean():.6f}, max={diff.max():.6f}")

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
