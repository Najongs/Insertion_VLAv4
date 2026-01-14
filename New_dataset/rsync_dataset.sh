# rsync -avP -e 'ssh -p [포트번호]' [보낼파일] [계정]@[IP]:[경로]
rsync -avP -e 'ssh -p 51322 -c aes128-ctr' /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data najo@10.130.4.79:/home/najo/NAS/VLA/dataset/New_dataset
rsync -avP -e ssh /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_new_dataset_ddp/checkpoints/checkpoint_step_25000.pt irom@10.130.41.45:/home/irom/NAS/VLA/Insertion_VLAv4/Inference/checkpoints


scp -P 51322 -r /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data najo@10.130.4.79:/home/najo/NAS/VLA/dataset/New_dataset


cd /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data
python3 -m http.server 8888

# 서버의 받을 폴더로 이동
cd /home/najo/NAS/VLA/dataset/New_dataset
# wget으로 긁어오기 (IP주소 변경 필요!)
wget -r -np -R "index.html*" http://10.130.41.45:8888/
10.130.41.45


rsync -avP -e ssh /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion/checkpoints/checkpoint_step_4000.pt irom@10.130.41.45:/home/irom/NAS/VLA/Insertion_VLAv4/Inference

