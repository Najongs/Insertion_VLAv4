# rsync -avP -e 'ssh -p [포트번호]' [보낼파일] [계정]@[IP]:[경로]
rsync -avP -e 'ssh -p 51322' /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data najo@10.130.4.79:/home/najo/NAS/VLA/dataset/New_dataset
rsync -avP -e 'ssh -p 51322 -c aes128-ctr' /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data najo@10.130.4.79:/home/najo/NAS/VLA/dataset/New_dataset

rsync -avP -e 'ssh -p 51322' -c aes128-ctr' //home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_new_dataset_ddp/checkpoints irom@10.130.41.45:/home/irom/NAS/VLA/Insertion_VLAv4/Inference/checkpoints