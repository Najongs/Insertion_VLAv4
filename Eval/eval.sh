PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
    /home/najo/NAS/VLA/Insertion_VLAv4/Eval/evaluate_episode_normalized.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_new/checkpoints/checkpoint_step_50000.pt \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/episode_eval_ep50 \
    --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
    --task_instruction "Insert needle into eye trocar"

PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
    /home/najo/NAS/VLA/Insertion_VLAv4/Eval/evaluate_episode_normalized.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_new/checkpoints/checkpoint_step_46000.pt \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/episode_eval_ep46 \
    --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
    --task_instruction "Insert needle into eye trocar"

PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
    /home/najo/NAS/VLA/Insertion_VLAv4/Eval/evaluate_episode_normalized.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_new/checkpoints/checkpoint_step_44000.pt \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/episode_eval_ep44 \
    --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
    --task_instruction "Insert needle into eye trocar"

PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
    /home/najo/NAS/VLA/Insertion_VLAv4/Eval/evaluate_episode_normalized.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_new/checkpoints/checkpoint_step_40000.pt \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/episode_eval_ep40 \
    --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
    --task_instruction "Insert needle into eye trocar"

# PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
#     /home/najo/NAS/VLA/Insertion_VLAv4/Eval/evaluate_episode_normalized.py \
#     --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_new/checkpoints/checkpoint_step_12000.pt \
#     --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
#     --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/episode_eval_ep12 \
#     --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
#     --task_instruction "Insert needle into eye trocar"