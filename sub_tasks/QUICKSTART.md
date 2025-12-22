# Hugging Face 모델 업로드 - Quick Start Guide

## 5분 만에 시작하기

### Step 1: Hugging Face 토큰 설정

```bash
# Hugging Face 토큰 설정 (https://huggingface.co/settings/tokens 에서 생성)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"
```

### Step 2: Repository ID 수정

`upload_model.sh` 파일을 열어서 본인의 username으로 변경:

```bash
nano upload_model.sh

# 다음 라인 수정:
REPO_ID="username/smolvla-insertion-vla"  # username을 본인의 Hugging Face username으로 변경
```

### Step 3: 업로드 실행

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/sub_tasks
bash upload_model.sh
```

끝! 🎉

---

## 체크리스트

- [ ] Hugging Face 계정 생성
- [ ] Write 권한이 있는 API 토큰 생성
- [ ] `HF_TOKEN` 환경 변수 설정
- [ ] `upload_model.sh`에서 `REPO_ID` 수정
- [ ] 스크립트 실행

---

## 일반적인 문제

### Q: "Invalid token" 오류가 나요
**A:** 토큰이 **Write** 권한이 있는지 확인하세요. Settings → Access Tokens에서 확인.

### Q: Repository가 이미 존재한다고 나요
**A:** 괜찮습니다. 기존 repository에 업데이트됩니다.

### Q: 업로드가 너무 느려요
**A:** 모델 크기에 따라 시간이 걸릴 수 있습니다. 네트워크 속도를 확인하세요.

---

## 다음 단계

업로드가 완료되면:

1. https://huggingface.co/username/smolvla-insertion-vla 에서 모델 확인
2. 모델 카드(README.md) 확인 및 필요시 수정
3. 다른 사람들과 공유!

## 모델 사용 예제

업로드된 모델을 사용하려면:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 모델 로드
policy = SmolVLAPolicy.from_pretrained("username/smolvla-insertion-vla")
policy.eval()

# 추론
action = policy.select_action(observation)
```

---

더 자세한 정보는 [README.md](README.md)를 참조하세요.
