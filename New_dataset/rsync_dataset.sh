# rsync -avP -e 'ssh -p [포트번호]' [보낼파일] [계정]@[IP]:[경로]
rsync -avP -e 'ssh -p 51322 -c aes128-ctr' /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data najo@10.130.4.79:/home/najo/NAS/VLA/dataset/New_dataset

scp -P 51322 -r /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data najo@10.130.4.79:/home/najo/NAS/VLA/dataset/New_dataset

cd /home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data
python3 -m http.server 8888

# 서버의 받을 폴더로 이동
cd /home/najo/NAS/VLA/dataset/New_dataset
# wget으로 긁어오기 (IP주소 변경 필요!)
wget -r -np -R "index.html*" http://10.130.41.45:8888/
10.130.41.45