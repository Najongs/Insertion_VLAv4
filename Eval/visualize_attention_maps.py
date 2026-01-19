#!/usr/bin/env python
"""
Visualize Vision Encoder Attention Maps for SmolVLA Model

This script extracts and visualizes attention maps from the vision encoder
for each frame of an episode, helping understand what the model focuses on.

Usage:
PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
    /home/najo/NAS/VLA/Insertion_VLAv4/Eval/visualize_attention_maps.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_new/checkpoints/checkpoint_step_10000.pt \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/attention_maps \
    --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
    --task_instruction "Insert needle into eye trocar" \
    --num_frames 50 \
    --cameras camera1 camera2 camera3
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import io
from matplotlib import cm

# Add Train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging
from normalization_utils import Normalizer, load_stats

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device):
    """Load model checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy_state_dict = checkpoint.get("policy_state_dict")
    if policy_state_dict is None:
        raise ValueError("Checkpoint does not contain 'policy_state_dict'")

    train_config = checkpoint.get("config")
    if train_config is None:
        logger.warning("Checkpoint does not contain training config")
        train_config = {}

    logger.info(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")

    policy_cfg = train_config.get("policy", {})
    pretrained_model_id = policy_cfg.get("pretrained_model_id", "lerobot/smolvla_base")

    logger.info(f"Loading base policy from: {pretrained_model_id}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_model_id)

    policy.config.n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    policy.config.chunk_size = policy_cfg.get("chunk_size", 1)
    policy.config.n_action_steps = policy_cfg.get("n_action_steps", 1)

    logger.info("Loading trained weights...")
    policy.load_state_dict(policy_state_dict, strict=False)

    policy.to(device)
    policy.eval()
    logger.info("Model loaded successfully")

    return policy, checkpoint


def load_episode_data(h5_path: str, num_frames: Optional[int] = None):
    """Load data from HDF5 episode file."""
    logger.info(f"Loading episode from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        images = {}
        for cam_key in ['camera1', 'camera2', 'camera3']:
            if cam_key in f['observations']['images']:
                cam_data = f['observations']['images'][cam_key][:]
                if num_frames is not None:
                    cam_data = cam_data[:num_frames]
                images[cam_key] = cam_data

        actions = f['action'][:]
        if num_frames is not None:
            actions = actions[:num_frames]

        ee_pose = f['observations']['ee_pose'][:]
        if num_frames is not None:
            ee_pose = ee_pose[:num_frames]

        timestamps = f['timestamp'][:]
        if num_frames is not None:
            timestamps = timestamps[:num_frames]

        actual_frames = len(actions)
        logger.info(f"Loaded {actual_frames} frames")

    return {
        'images': images,
        'actions': actions,
        'ee_pose': ee_pose,
        'timestamps': timestamps,
        'num_frames': actual_frames,
    }


def decode_image(img_bytes):
    """Decode JPEG bytes to PIL Image."""
    return Image.open(io.BytesIO(img_bytes))


class FeatureMapHook:
    """Hook to capture feature maps from vision encoder."""

    def __init__(self):
        self.feature_maps = []

    def __call__(self, module, input, output):
        """Hook function to capture feature maps."""
        # Store the output feature maps
        if isinstance(output, torch.Tensor):
            self.feature_maps.append(output.detach())
        elif isinstance(output, tuple):
            # Sometimes output is a tuple, take the first element
            self.feature_maps.append(output[0].detach() if isinstance(output[0], torch.Tensor) else output)

    def clear(self):
        """Clear stored feature maps."""
        self.feature_maps = []


class AttentionWeightsHook:
    """Hook to capture attention weights from transformer layers."""

    def __init__(self):
        self.attention_weights = []

    def __call__(self, module, input, output):
        """Hook function to capture attention weights."""
        # For transformer attention layers, output is typically (hidden_states, attention_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]  # Attention weights
            if attn is not None and isinstance(attn, torch.Tensor):
                self.attention_weights.append(attn.detach())

    def clear(self):
        """Clear stored attention weights."""
        self.attention_weights = []


def get_vision_encoder_features(model, observation):
    """
    Extract vision encoder feature maps for each camera.
    Uses forward hook to capture intermediate activations.
    """
    attention_maps = {}

    # Access the VLM model: policy.model.vlm_with_expert.vlm.model.vision_model
    if not hasattr(model, 'model'):
        logger.error("Model does not have 'model' attribute")
        return attention_maps

    vlm_flow_model = model.model
    if not hasattr(vlm_flow_model, 'vlm_with_expert'):
        logger.error("Model does not have 'vlm_with_expert' attribute")
        return attention_maps

    vlm_with_expert = vlm_flow_model.vlm_with_expert

    if not hasattr(vlm_with_expert, 'vlm'):
        logger.error("vlm_with_expert does not have 'vlm' attribute")
        return attention_maps

    vlm = vlm_with_expert.vlm

    # Access the vision model through vlm.model.vision_model
    if not hasattr(vlm, 'model'):
        logger.error("vlm does not have 'model' attribute")
        return attention_maps

    vlm_model = vlm.model
    if not hasattr(vlm_model, 'vision_model'):
        logger.error("vlm.model does not have 'vision_model' attribute")
        return attention_maps

    vision_model = vlm_model.vision_model
    logger.info(f"Found vision model: {type(vision_model).__name__}")

    # Register hooks on attention layers to capture attention weights
    attention_hooks = []
    attn_hook_handlers = []

    # Try to register on each attention layer
    if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
        # Register on the last attention layer for best semantic information
        layers = vision_model.encoder.layers
        if len(layers) > 0:
            last_layer = layers[-1]  # Use last layer
            if hasattr(last_layer, 'self_attn'):
                attn_hook = AttentionWeightsHook()
                handle = last_layer.self_attn.register_forward_hook(attn_hook)
                attention_hooks.append(attn_hook)
                attn_hook_handlers.append(handle)
                logger.info(f"Registered attention hook on last encoder layer")

    # Process each camera separately
    for cam_key in ['camera1', 'camera2', 'camera3']:
        obs_key = f'observation.images.{cam_key}'
        if obs_key not in observation:
            continue

        img_tensor = observation[obs_key]

        # Register hook on vision encoder for feature maps
        feature_hook = FeatureMapHook()
        hook_handle = vision_model.register_forward_hook(feature_hook)

        # Clear attention hooks
        for attn_hook in attention_hooks:
            attn_hook.clear()

        try:
            # Run forward pass through vision encoder
            with torch.no_grad():
                # Call vision model directly with the image
                # SmolVLM vision model expects [B, C, H, W] input
                # Match the dtype of the vision model
                img_tensor_dtype = img_tensor.to(vision_model.embeddings.patch_embedding.weight.dtype)
                output = vision_model(img_tensor_dtype, output_attentions=True)

            # Extract attention/feature map
            attn_map_final = None

            # Method 1: Use ACTUAL attention weights (not feature magnitude)
            if hasattr(output, 'attentions') and output.attentions is not None:
                attentions = output.attentions

                if isinstance(attentions, tuple) and len(attentions) > 0:
                    # Try different approaches to get meaningful attention maps

                    # Approach 1: Use MIDDLE layers instead of last layer
                    # Last layers often have very diffuse attention
                    # Middle layers capture more specific visual features
                    num_layers = len(attentions)
                    middle_layer_idx = num_layers // 2

                    logger.info(f"{cam_key}: Trying multiple attention layers ({num_layers} total)")

                    # Collect attention from multiple layers
                    all_strategies = []

                    for layer_idx in [middle_layer_idx - 1, middle_layer_idx, middle_layer_idx + 1]:
                        if 0 <= layer_idx < num_layers:
                            layer_attn = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]

                            # Try different attention heads separately
                            num_heads = layer_attn.shape[0]
                            for head_idx in range(num_heads):
                                head_attn = layer_attn[head_idx]  # [seq_len, seq_len]

                                # CLS attention from this head
                                cls_attn = head_attn[0, 1:]

                                var = cls_attn.std().item()
                                all_strategies.append({
                                    'name': f'L{layer_idx}H{head_idx}',
                                    'values': cls_attn,
                                    'variance': var
                                })

                    # Select the head with highest variance
                    if all_strategies:
                        best = max(all_strategies, key=lambda x: x['variance'])
                        patch_importance = best['values']

                        logger.info(f"  Best: {best['name']} (std: {best['variance']:.6f})")
                        logger.info(f"  Values - min: {patch_importance.min():.4f}, max: {patch_importance.max():.4f}")
                else:
                    logger.warning(f"{cam_key}: No attention weights available, falling back to features")
                    patch_importance = None
            else:
                patch_importance = None

            # Fallback to feature magnitude if attention not available
            if patch_importance is None and hasattr(output, 'last_hidden_state'):
                last_hidden_state = output.last_hidden_state
                patch_features = last_hidden_state[0, 1:, :]
                patch_importance = patch_features.float().abs().mean(dim=-1)

                num_patches = patch_importance.shape[0]
                logger.info(f"{cam_key}: Using feature magnitude fallback")
                logger.info(f"  Shape: {patch_importance.shape}, min: {patch_importance.min():.4f}, max: {patch_importance.max():.4f}")

            if patch_importance is not None:
                num_patches = patch_importance.shape[0]

                # SmolVLM uses image splitting: 1 global + 2 local crops
                # 1199 = 399*3 + 2 special tokens
                # Each patch should be 399 = 21x19 tokens

                if num_patches == 1199:
                    logger.info(f"  Detected SmolVLM image splitting (1199 tokens)")

                    # Remove special tokens and split into 3 patches
                    # Assume 2 special tokens at specific positions
                    visual_tokens = patch_importance[:-2] if num_patches == 1199 else patch_importance

                    chunk_size = len(visual_tokens) // 3  # 399

                    # Split into 3 patches: Global + 2 Local Crops
                    patches = []
                    for i in range(3):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size
                        patch_chunk = visual_tokens[start:end]
                        patches.append(patch_chunk)

                    # Use global patch (first one, typically has full image view)
                    global_patch = patches[0]

                    # Normalize
                    global_patch = (global_patch - global_patch.min()) / (global_patch.max() - global_patch.min() + 1e-8)

                    # 399 = 21 x 19 (closest to square)
                    h, w = 21, 19
                    logger.info(f"  Using global patch, reshaping to {h}x{w}")

                    try:
                        attn_map_2d = global_patch.reshape(h, w)

                        # Interpolate to larger size for better visualization
                        target_size = 28  # Larger for better detail
                        attn_map_resized = F.interpolate(
                            attn_map_2d.unsqueeze(0).unsqueeze(0).float(),
                            size=(target_size, target_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().cpu().numpy()

                        attn_map_final = attn_map_resized
                        logger.info(f"  Final attention map shape: {attn_map_final.shape}")
                    except Exception as e:
                        logger.warning(f"  Failed to reshape: {e}")
                        # Fallback to old method
                        pass

                # Fallback for other configurations
                if attn_map_final is None:
                    # Normalize
                    patch_importance = (patch_importance - patch_importance.min()) / (patch_importance.max() - patch_importance.min() + 1e-8)

                    # Try to find best rectangular grid
                    best_h, best_w = 1, num_patches
                    for h in range(1, int(np.sqrt(num_patches)) + 10):
                        if num_patches % h == 0:
                            w = num_patches // h
                            if abs(h - w) < abs(best_h - best_w):
                                best_h, best_w = h, w

                    logger.info(f"  Fallback: reshaping {num_patches} patches to {best_h}x{best_w}")

                    attn_map_2d = patch_importance.reshape(best_h, best_w)
                    target_size = 14
                    attn_map_resized = F.interpolate(
                        attn_map_2d.unsqueeze(0).unsqueeze(0).float(),
                        size=(target_size, target_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()

                    attn_map_final = attn_map_resized
                    logger.info(f"  Final attention map shape: {attn_map_final.shape}")

            if attn_map_final is not None:
                attention_maps[cam_key] = attn_map_final
                continue

            # Otherwise, fall back to feature map approach
            if len(feature_hook.feature_maps) > 0:
                features = feature_hook.feature_maps[0]
                logger.info(f"Captured features for {cam_key}: shape {features.shape}, type {type(features)}")

                # SmolVLM vision outputs are typically tuples (last_hidden_state, ...)
                if isinstance(features, tuple):
                    features = features[0]  # Take last_hidden_state

                # Convert features to attention-like map
                # Features typically have shape [B, num_patches, hidden_dim]
                if features.ndim == 3:
                    # Average across hidden dimension
                    attn_map = features[0].abs().mean(dim=-1)  # [num_patches]

                    # Normalize
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                    # Reshape to 2D (assuming square patch grid)
                    num_patches = attn_map.shape[0]
                    grid_size = int(np.sqrt(num_patches))

                    if grid_size * grid_size == num_patches:
                        attn_map = attn_map.reshape(grid_size, grid_size)
                    else:
                        # Try common sizes for vision transformers
                        # 224x224 image with 16x16 patches = 14x14 grid
                        # But may have class token, so could be 196+1=197 patches
                        if num_patches == 197:  # 14x14 + 1 class token
                            attn_map = attn_map[1:]  # Remove class token
                            attn_map = attn_map.reshape(14, 14)
                        elif num_patches == 256:  # 16x16 patches
                            attn_map = attn_map.reshape(16, 16)
                        else:
                            # Try to make it work
                            h = int(np.sqrt(num_patches))
                            w = num_patches // h
                            if h * w == num_patches:
                                attn_map = attn_map.reshape(h, w)
                            else:
                                logger.warning(f"Cannot reshape {num_patches} patches to square grid")
                                continue

                    attention_maps[cam_key] = attn_map.cpu().numpy()
                    logger.info(f"Extracted feature map for {cam_key}: shape {attn_map.shape}")

                elif features.ndim == 4:
                    # CNN-style features [B, C, H, W]
                    attn_map = features[0].abs().mean(dim=0)  # Average over channels [H, W]
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    attention_maps[cam_key] = attn_map.cpu().numpy()
                    logger.info(f"Extracted CNN feature map for {cam_key}: shape {attn_map.shape}")
                else:
                    logger.warning(f"Unexpected feature shape: {features.shape}")

        except Exception as e:
            logger.error(f"Error extracting features for {cam_key}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Remove hook
            hook_handle.remove()
            feature_hook.clear()

    # Remove attention hooks
    for handle in attn_hook_handlers:
        handle.remove()

    return attention_maps


def prepare_observation(
    episode_data,
    frame_idx,
    device,
    policy,
    normalizer,
    task_instruction="Insert needle into target point"
):
    """Prepare observation for model input with normalization."""
    # Decode images
    images_list = []
    for cam_key in ['camera1', 'camera2', 'camera3']:
        if cam_key in episode_data['images']:
            img_bytes = episode_data['images'][cam_key][frame_idx]
            img = decode_image(img_bytes)
            images_list.append((cam_key, img))

    # Get state
    state = episode_data['ee_pose'][frame_idx]
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    state_normalized = normalizer.normalize(state_tensor, 'observation.state')

    observation = {
        'observation.state': state_normalized,
    }

    # Add images
    for cam_key, img in images_list:
        observation[f'observation.images.{cam_key}'] = img

    # Add language tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    tokens = tokenizer(
        task_instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    observation['observation.language.tokens'] = tokens['input_ids'].squeeze(0)
    observation['observation.language.attention_mask'] = tokens['attention_mask'].squeeze(0)

    # Move to device and add batch dimension
    batch = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, Image.Image):
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            batch[key] = to_tensor(value).unsqueeze(0).to(device)

    return batch


def visualize_attention_on_image(image, attention_map, alpha=0.6):
    """
    Overlay attention map on image.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        attention_map: numpy array (H, W) with values in [0, 1]
        alpha: transparency of overlay (higher = more attention color visible)

    Returns:
        numpy array with attention overlay
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Resize attention map to match image size
    h, w = image.shape[:2]
    attention_resized = F.interpolate(
        torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float(),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Enhance contrast - apply power transform to make differences more visible
    # Values closer to 0 become even smaller, values closer to 1 stay high
    attention_enhanced = np.power(attention_resized, 0.7)  # Gamma correction

    # Apply colormap - use 'hot' or 'viridis' for better visibility
    colormap = cm.get_cmap('hot')  # Red-yellow colormap, easier to see
    attention_colored = colormap(attention_enhanced)[:, :, :3]  # Remove alpha channel
    attention_colored = (attention_colored * 255).astype(np.uint8)

    # Blend with original image - use weighted blend
    # Higher alpha means more attention color, less original image
    blended = (alpha * attention_colored + (1 - alpha) * image).astype(np.uint8)

    return blended, attention_resized


def extract_and_visualize_frame(
    policy,
    episode_data,
    frame_idx,
    normalizer,
    device,
    task_instruction,
    output_dir,
    selected_cameras
):
    """Extract attention maps for a single frame and visualize."""
    logger.info(f"Processing frame {frame_idx}")

    # Prepare observation
    observation = prepare_observation(
        episode_data,
        frame_idx,
        device,
        policy,
        normalizer,
        task_instruction
    )

    # Get attention maps using feature map extraction
    attention_maps = get_vision_encoder_features(policy, observation)

    # Visualize
    num_cameras = len(selected_cameras)
    fig, axes = plt.subplots(2, num_cameras, figsize=(6 * num_cameras, 10))

    if num_cameras == 1:
        axes = axes.reshape(-1, 1)

    for idx, cam_key in enumerate(selected_cameras):
        if cam_key not in episode_data['images']:
            continue

        # Get original image
        img_bytes = episode_data['images'][cam_key][frame_idx]
        img = decode_image(img_bytes)
        img_np = np.array(img)

        # Plot original image
        axes[0, idx].imshow(img_np)
        axes[0, idx].set_title(f'{cam_key} - Original')
        axes[0, idx].axis('off')

        # Plot attention overlay
        if cam_key in attention_maps:
            attention_map = attention_maps[cam_key]
            logger.info(f"  {cam_key} attention - min: {attention_map.min():.4f}, max: {attention_map.max():.4f}, mean: {attention_map.mean():.4f}")
            blended, attn_resized = visualize_attention_on_image(img_np, attention_map, alpha=0.6)
            axes[1, idx].imshow(blended)
            axes[1, idx].set_title(f'{cam_key} - Attention Overlay')
        else:
            axes[1, idx].imshow(img_np)
            axes[1, idx].set_title(f'{cam_key} - No Attention')
        axes[1, idx].axis('off')

    plt.suptitle(f'Frame {frame_idx} - Vision Encoder Attention', fontsize=16)
    plt.tight_layout()

    # Save
    output_path = output_dir / f'frame_{frame_idx:05d}_attention.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved attention visualization to: {output_path}")

    return attention_maps


def main():
    parser = argparse.ArgumentParser(description="Visualize Vision Encoder Attention Maps")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)"
    )

    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Path to HDF5 episode file"
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to dataset statistics YAML file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/attention_maps",
        help="Output directory for visualizations"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--task_instruction",
        type=str,
        default="Insert needle into target point",
        help="Task instruction text"
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=50,
        help="Number of frames to process (default: 50)"
    )

    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, process all)"
    )

    parser.add_argument(
        "--cameras",
        nargs='+',
        default=['camera1', 'camera2', 'camera3'],
        help="Cameras to visualize (default: all three)"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    episode_path = Path(args.episode)
    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not episode_path.exists():
        logger.error(f"Episode not found: {episode_path}")
        sys.exit(1)

    if not stats_path.exists():
        logger.error(f"Statistics file not found: {stats_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = get_safe_torch_device(args.device)
    logger.info(f"Using device: {device}")

    # Load normalization statistics
    logger.info(f"Loading normalization statistics...")
    stats = load_stats(str(stats_path))
    normalizer = Normalizer(stats).to(device)

    # Load checkpoint
    policy, checkpoint = load_checkpoint(str(checkpoint_path), device)

    # Load episode data
    episode_data = load_episode_data(str(episode_path), num_frames=args.num_frames)

    # Process frames
    logger.info(f"Processing {episode_data['num_frames']} frames...")
    logger.info(f"Cameras: {args.cameras}")
    logger.info(f"Frame skip: {args.frame_skip}")

    for frame_idx in range(0, episode_data['num_frames'], args.frame_skip):
        try:
            attention_maps = extract_and_visualize_frame(
                policy=policy,
                episode_data=episode_data,
                frame_idx=frame_idx,
                normalizer=normalizer,
                device=device,
                task_instruction=args.task_instruction,
                output_dir=output_dir,
                selected_cameras=args.cameras
            )
        except Exception as e:
            logger.error(f"Failed to process frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\nAll visualizations saved to: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
