# Model Comparison: SmolVLA vs ACT vs Diffusion Policy

## Overview

This document compares three policies for needle insertion task:
- **SmolVLA**: Vision-Language-Action model (current baseline)
- **ACT**: Action Chunking Transformer (recommended for insertion)
- **Diffusion Policy**: Diffusion-based policy (highest precision)

---

## Quick Comparison Table

| Feature | SmolVLA | ACT | Diffusion Policy |
|---------|---------|-----|------------------|
| **Best For** | General manipulation | Bimanual insertion | Tight-tolerance tasks |
| **Pretrained** | ✅ smolvla_base | ✅ act_aloha_sim_insertion_human | ⚠️ diffusion_pusht (2D only) |
| **Action Chunking** | ✅ Yes (10) | ✅ Yes (100) | ✅ Yes (16) |
| **Training Steps** | 25k-50k | 50k | 100k |
| **Training Time** | ~3 hours | ~5 hours | ~10 hours |
| **Inference Speed** | Fast (8 Hz) | Fast (15 Hz) | Slow (5 Hz, diffusion sampling) |
| **Sample Efficiency** | Medium | High | Medium |
| **Precision** | Medium | High | Very High |
| **Success Rate** | ? (untested) | 80-90% (50 demos) | 95.7% (TacDiffusion) |
| **Model Size** | Large (VLM) | Small (80M) | Medium |
| **Language** | ✅ Yes | ❌ No | ❌ No |

---

## Detailed Comparison

### 1. SmolVLA (Current Baseline)

**Architecture:**
- Vision-Language-Action model
- Pre-trained on SO-100 (simulated tabletop manipulation)
- Frozen VLM + trained action expert

**Strengths:**
- ✅ Language-conditioned (can follow text instructions)
- ✅ General-purpose manipulation
- ✅ Pre-trained on diverse tasks

**Weaknesses:**
- ⚠️ SO-100 bias (simulation → real gap)
- ⚠️ Less proven for precise insertion
- ⚠️ Large model (slow inference potential)

**When to Use:**
- Multi-task scenarios with language instructions
- Tabletop manipulation
- When you need VLM capabilities

**Training Command:**
```bash
cd Train
bash train_ddp.sh
```

---

### 2. ACT (Action Chunking Transformer) **⭐ RECOMMENDED**

**Architecture:**
- Transformer-based encoder-decoder
- Pre-trained on ALOHA peg insertion (similar to needle insertion!)
- CVaE for action generation

**Strengths:**
- ✅ **Proven on insertion tasks** (ALOHA peg-in-hole)
- ✅ **Sample efficient** (50 demos → 80% success)
- ✅ Fast training and inference
- ✅ Lightweight model (80M parameters)
- ✅ Temporal consistency (action chunking)

**Weaknesses:**
- ❌ No language conditioning
- ⚠️ Simulation→real gap (pretrained on sim)

**When to Use:**
- **Needle insertion** (recommended!)
- Bimanual manipulation
- Contact-rich tasks
- When you need fast inference

**Key Papers:**
- [Learning Fine-Grained Bimanual Manipulation](https://huggingface.co/papers/2304.13705)
- ALOHA: 80-90% success on insertion tasks

**Training Command:**
```bash
cd Train
bash train_act.sh
```

**Performance Expectations:**
- 50 demonstrations → ~80% success rate
- 100 demonstrations → ~90% success rate
- Trains in 3-5 hours (5 GPUs)
- Inference: 15 Hz (real-time capable)

---

### 3. Diffusion Policy **⭐ HIGHEST PRECISION**

**Architecture:**
- Diffusion model for action generation (DDPM/DDIM)
- Generates smooth trajectories via iterative denoising
- Vision encoder + diffusion transformer

**Strengths:**
- ✅ **Highest precision** (95.7% on tight-clearance insertion)
- ✅ **Smooth trajectories** (diffusion process)
- ✅ **Multi-modal** (handles uncertainty)
- ✅ Excellent for contact-rich manipulation

**Weaknesses:**
- ⚠️ Slower inference (diffusion sampling: 100 steps)
- ⚠️ Longer training time (100k steps)
- ⚠️ No insertion-specific pretrained model
- ❌ No language conditioning

**When to Use:**
- **High-precision insertion** (millimeter tolerance)
- Contact-rich manipulation
- When trajectory smoothness is critical
- When inference speed is not critical

**Key Papers:**
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [TacDiffusion](https://arxiv.org/html/2409.11047v1): 95.7% on tight-clearance insertion

**Training Command:**
```bash
cd Train
bash train_diffusion.sh
```

**Performance Expectations:**
- TacDiffusion: 95.7% on tight-clearance insertion
- Requires more data than ACT
- Trains in 8-10 hours (5 GPUs)
- Inference: 5-8 Hz (slower due to sampling)

---

## Recommendation for Needle Insertion

### **Priority 1: ACT** ⭐⭐⭐⭐⭐
- **Best balance** of precision, speed, and sample efficiency
- **Pretrained on insertion task** (`act_aloha_sim_insertion_human`)
- Fast training and inference
- Proven 80-90% success on peg insertion

### **Priority 2: Diffusion Policy** ⭐⭐⭐⭐
- **Highest precision** (95.7% on tight-clearance insertion)
- Best for millimeter-level tolerance
- Slower but smoother

### **Priority 3: SmolVLA** ⭐⭐
- Current baseline
- Less proven for insertion
- Good for general manipulation + language

---

## Training Workflow

### Step 1: Try ACT First (Fastest Path to Success)

```bash
# 1. Edit config (if needed)
vim train_config_act.yaml

# 2. Start training (5 GPUs, ~5 hours)
bash train_act.sh

# 3. Monitor logs
tail -f outputs/train/act_needle_insertion/logs/train_*.log

# 4. Evaluate
cd ../Eval
python evaluate_act.py
```

**Expected Result:** 80-90% success rate with current dataset

---

### Step 2: If ACT is Insufficient, Try Diffusion Policy

```bash
# 1. Edit config (if needed)
vim train_config_diffusion.yaml

# 2. Start training (5 GPUs, ~10 hours)
bash train_diffusion.sh

# 3. Monitor logs
tail -f outputs/train/diffusion_needle_insertion/logs/train_*.log

# 4. Evaluate
cd ../Eval
python evaluate_diffusion.py
```

**Expected Result:** 95%+ success rate (higher precision)

---

### Step 3: Compare with SmolVLA Baseline

```bash
# Already trained
cd Train/outputs/train/smolvla_new_dataset_ddp

# Evaluate all three
cd ../../Eval
python compare_models.py --models act diffusion smolvla
```

---

## Configuration Guide

### Key Hyperparameters

| Parameter | SmolVLA | ACT | Diffusion |
|-----------|---------|-----|-----------|
| `horizon` | 10 | 100 | 16 |
| `n_action_steps` | 10 | 100 | 8 |
| `batch_size` | 16 | 8 | 8 |
| `lr` | 1e-4 | 5e-5 | 1e-4 |
| `steps` | 25k | 50k | 100k |
| `augmentation` | Conservative | Aggressive | Very Aggressive |

### When to Adjust

**Overfitting (low train loss, poor eval):**
- ✅ Increase augmentation
- ✅ Increase horizon
- ✅ Add dropout

**Underfitting (high train loss):**
- ✅ Decrease horizon
- ✅ Increase model capacity
- ✅ Train longer

**Jittery actions:**
- ✅ Increase `horizon` (ACT)
- ✅ Enable `temporal_ensemble_coeff` (ACT)
- ✅ Already smooth (Diffusion)

---

## Troubleshooting

### Problem: ACT actions are jittery

**Solution 1:** Enable temporal ensemble
```yaml
# train_config_act.yaml
policy:
  temporal_ensemble_coeff: 0.01  # Light smoothing
```

**Solution 2:** Increase action chunk
```yaml
policy:
  chunk_size: 150  # Increase from 100
  n_action_steps: 150
```

---

### Problem: Diffusion inference too slow

**Solution:** Reduce inference steps (trade quality for speed)
```yaml
policy:
  num_inference_steps: 20  # Reduce from 100 (still good quality)
```

---

### Problem: Not enough training data

**ACT:** Still works with 20-50 demos (sample efficient)
**Diffusion:** Needs 50-100 demos for best results
**SmolVLA:** Needs 100+ demos

---

## Next Steps

1. **Start with ACT** (fastest path to success)
2. **Compare with SmolVLA** (current baseline)
3. **Try Diffusion** if precision is insufficient
4. **Collect more data** if success rate < 80%
5. **Tune hyperparameters** for your specific task

---

## References

- [ACT Paper](https://huggingface.co/papers/2304.13705)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [TacDiffusion Paper](https://arxiv.org/html/2409.11047v1)
- [SmolVLA Paper](https://huggingface.co/papers/2506.01844)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
