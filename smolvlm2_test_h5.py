"""
Test SmolVLM2 with images from HDF5 episode file
Loads one frame with 3 camera images and processes them with a text prompt
"""
import h5py
import os
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def load_frame_from_h5(h5_path, frame_idx=0):
    """
    Load images from one frame of an HDF5 episode file

    Args:
        h5_path: Path to HDF5 file
        frame_idx: Index of frame to load (default: 0)

    Returns:
        list of PIL Images (3 images from camera1, camera2, camera3)
    """
    images = []

    with h5py.File(h5_path, 'r') as f:
        # Load images from each camera
        for cam_key in ['camera1', 'camera2', 'camera3']:
            # Get compressed JPEG bytes
            img_bytes = f['observations']['images'][cam_key][frame_idx]

            # Decode JPEG to PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

            print(f"Loaded {cam_key}: {img.size} pixels, mode={img.mode}")

    return images


def main():
    # Configuration
    h5_path = "/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260106/1_MIN/episode_20260106_130754_trimmed_0_323.h5"
    frame_idx = 250  # Which frame to use (0 to 322)

    # Text prompt for the model
    text_prompt = """You are analyzing a surgical robotic system performing a needle insertion task into an eye model.
Please carefully examine these three camera views and provide a detailed analysis:

1. **Needle Detection**: Identify and describe the needle's current position and orientation. Where is the needle located relative to the eye model and the target insertion point?

2. **Current Action**: What is the robot currently doing? Describe the specific manipulation or movement being performed (e.g., approaching, grasping, positioning, inserting, retracting).

3. **Progress Assessment**: At what stage of the insertion task does this appear to be? (e.g., initial approach, pre-insertion positioning, active insertion, post-insertion, task completion)

4. **Next Required Action**: Based on the current state, what should the robot do next to successfully complete the needle insertion task?

5. **Spatial Relationships**: Describe the relative positions of key elements - the needle, the eye model, the robot gripper, and any visible target markers.

Please be specific and technical in your analysis."""

    print(f"Loading frame {frame_idx} from {h5_path}")
    print()

    # Load images from HDF5
    images = load_frame_from_h5(h5_path, frame_idx)
    print()

    # Save images to sample directory
    sample_dir = "/home/najo/NAS/VLA/Insertion_VLAv4/sample"
    os.makedirs(sample_dir, exist_ok=True)

    print(f"Saving images to {sample_dir}...")
    for idx, (img, cam_name) in enumerate(zip(images, ['camera1', 'camera2', 'camera3'])):
        save_path = os.path.join(sample_dir, f"frame_{frame_idx}_{cam_name}.jpg")
        img.save(save_path, quality=95)
        print(f"  Saved {save_path}")
    print()

    # Load model and processor
    print("Loading SmolVLM2 model...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2"  # Commented out - requires flash-attn
    ).to("cuda")
    print("Model loaded successfully")
    print()

    # Prepare messages with all 3 images
    # SmolVLM2 can handle multiple images in one message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # camera1
                {"type": "image"},  # camera2
                {"type": "image"},  # camera3
                {"type": "text", "text": text_prompt}
            ]
        },
    ]

    # Process inputs - pass images as a list
    print("Processing inputs...")
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
        images=images,  # List of 3 PIL Images
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # Generate response
    print("Generating response...")
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # Print result
    print("=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(generated_texts[0])
    print("=" * 80)


if __name__ == "__main__":
    main()
