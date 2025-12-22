---
license: apache-2.0
tags:
- robotics
- vision-language-action
- smolvla
- lerobot
- insertion-task
library_name: lerobot
pipeline_tag: robotics
---

# SmolVLA for VLA Insertion Task

This model is a fine-tuned version of [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
on a robot insertion task dataset with multi-camera views and sensor feedback.

## Model Description

**SmolVLA (Small Vision-Language-Action)** is a compact vision-language-action model designed for robot manipulation tasks.
This checkpoint has been trained on a custom VLA insertion dataset with 5 different colored insertion points.

- **Base Model:** lerobot/smolvla_base
- **Training Framework:** LeRobot
- **Task:** Precision needle insertion with visual and force feedback
- **Robot:** Meca500 (6-DOF collaborative robot)

## Training Details

### Training Data

- **Dataset:** VLA Insertion Task Dataset
- **Episodes:** 419 demonstrations
- **Colors:** 5 insertion targets (Blue, Green, Red, White, Yellow)
- **Observations:**
  - 5 camera views (640x480 RGB images)
  - Robot state (6-DOF pose)
  - OCT sensor data (1025 A-lines)
  - Force feedback
- **Actions:** 6-DOF delta pose commands (position + rotation)

### Training Configuration

```yaml
Training Steps: 16000
Epochs: unknown
Observation Steps: 1
Action Chunk Size: 1
Action Prediction Steps: 1
```

### Training Procedure

The model was trained using:
- **Optimizer:** AdamW
- **Learning Rate:** from training config
- **Hardware:** Multi-GPU training with DataParallel
- **Framework:** LeRobot + PyTorch

## Intended Use

This model is intended for:
- Research in vision-language-action models
- Robot manipulation with visual feedback
- Precision insertion tasks
- Multi-modal robot learning

### Example Usage

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Load the trained model
model_id = "Najongs/smolvla-insertion-vla"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.eval()

# Prepare observation
observation = {
    "observation.images.camera1": image_tensor,  # Shape: (1, 3, 480, 640)
    "observation.state": state_tensor,           # Shape: (1, 6)
    "task": "Insert needle into Red point",
    "robot_type": "meca500",
}

# Get action prediction
with torch.no_grad():
    action = policy.select_action(observation)

print(f"Predicted action: {action}")  # Shape: (1, 6) - delta pose
```

## Model Architecture

SmolVLA combines:
1. **Vision Encoder:** Processes multi-camera RGB images
2. **Language Encoder:** Processes task instructions
3. **Action Decoder:** Predicts robot actions from visual and language inputs

Key features:
- Multi-camera fusion
- Vision-language alignment
- Action chunking for temporal consistency
- Efficient architecture for real-time inference

## Limitations and Biases

- Trained specifically for insertion tasks with the Meca500 robot
- Performance may vary with different lighting conditions
- Requires similar camera setup (5 views) for best results
- Limited to the insertion target colors seen during training

## Training Infrastructure

- **Framework:** LeRobot + PyTorch
- **Hardware:** Multi-GPU setup
- **Checkpoint:** checkpoint_step_0016000.pt

## Citation

If you use this model, please cite:

```bibtex
@misc{smolvla-insertion-task,
  title={SmolVLA for VLA Insertion Task},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/Najongs/smolvla-insertion-vla}},
}
```

## Model Card Authors

Created by the VLA Insertion Task team.

## Model Card Contact

For questions or issues, please open an issue in the repository.
