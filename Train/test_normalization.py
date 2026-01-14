#!/usr/bin/env python
"""
Test normalization functionality.

This script tests that normalization and unnormalization work correctly.
"""

import torch
import numpy as np
from normalization_utils import Normalizer, load_stats

def test_normalization():
    """Test normalization round-trip."""
    print("="*80)
    print("TESTING NORMALIZATION")
    print("="*80)

    # Load stats
    stats_path = "dataset_stats.yaml"
    print(f"\nLoading stats from: {stats_path}")
    stats = load_stats(stats_path)

    # Create normalizer
    normalizer = Normalizer(stats)
    print("Normalizer created\n")

    # Test action normalization
    print("-" * 80)
    print("TEST 1: Action Normalization")
    print("-" * 80)

    action_mean = torch.tensor(stats['action']['mean'])
    action_std = torch.tensor(stats['action']['std'])

    # Create test action (use mean value)
    test_action = action_mean.clone()
    print(f"Original action:    {test_action}")

    # Normalize
    normalized_action = normalizer.normalize(test_action, 'action')
    print(f"Normalized action:  {normalized_action}")
    print(f"Expected (all ~0):  [0, 0, 0, 0, 0, 0]")

    # Unnormalize
    unnormalized_action = normalizer.unnormalize(normalized_action, 'action')
    print(f"Unnormalized action: {unnormalized_action}")
    print(f"Difference:          {torch.abs(unnormalized_action - test_action)}")

    # Check round-trip
    if torch.allclose(test_action, unnormalized_action, atol=1e-6):
        print("✓ Action round-trip successful!")
    else:
        print("✗ Action round-trip failed!")

    # Test state normalization
    print("\n" + "-" * 80)
    print("TEST 2: State Normalization")
    print("-" * 80)

    state_mean = torch.tensor(stats['observation.state']['mean'])
    state_std = torch.tensor(stats['observation.state']['std'])

    # Create test state (use mean value)
    test_state = state_mean.clone()
    print(f"Original state:     {test_state}")

    # Normalize
    normalized_state = normalizer.normalize(test_state, 'observation.state')
    print(f"Normalized state:   {normalized_state}")
    print(f"Expected (all ~0):  [0, 0, 0, 0, 0, 0]")

    # Unnormalize
    unnormalized_state = normalizer.unnormalize(normalized_state, 'observation.state')
    print(f"Unnormalized state:  {unnormalized_state}")
    print(f"Difference:          {torch.abs(unnormalized_state - test_state)}")

    # Check round-trip
    if torch.allclose(test_state, unnormalized_state, atol=1e-6):
        print("✓ State round-trip successful!")
    else:
        print("✗ State round-trip failed!")

    # Test with extreme values
    print("\n" + "-" * 80)
    print("TEST 3: Extreme Values")
    print("-" * 80)

    # Test with min/max values
    action_min = torch.tensor(stats['action']['min'])
    action_max = torch.tensor(stats['action']['max'])

    normalized_min = normalizer.normalize(action_min, 'action')
    normalized_max = normalizer.normalize(action_max, 'action')

    print(f"Action min normalized: {normalized_min[:3]}...")
    print(f"Action max normalized: {normalized_max[:3]}...")

    # Unnormalize
    recovered_min = normalizer.unnormalize(normalized_min, 'action')
    recovered_max = normalizer.unnormalize(normalized_max, 'action')

    if torch.allclose(action_min, recovered_min, atol=1e-5) and \
       torch.allclose(action_max, recovered_max, atol=1e-5):
        print("✓ Extreme values round-trip successful!")
    else:
        print("✗ Extreme values round-trip failed!")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    test_normalization()
