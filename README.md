# CartPole-v1 DQN 강화학습 프로젝트

OpenAI Gymnasium의 **CartPole-v1** 환경에서 **DQN(Deep Q-Network)** 알고리즘을 사용하여 강화학습 에이전트를 학습시키는 프로젝트입니다.

---

## 📋 프로젝트 개요

CartPole 문제는 막대가 카트 위에 수직으로 서 있는 상태를 유지하도록 카트를 좌우로 움직이는 고전적인 제어 문제입니다. 이 프로젝트는 딥러닝 기반 강화학습 알고리즘인 DQN을 사용하여 에이전트가 스스로 학습하도록 구현되었습니다.

### 주요 특징
- ✅ **DQN 알고리즘 구현**: Deep Q-Network를 사용한 가치 기반 강화학습
- ✅ **Experience Replay**: Replay Buffer를 통한 안정적인 학습
- ✅ **Target Network**: 학습 안정성을 위한 타겟 네트워크 사용
- ✅ **재현성 보장**: Random Seed 고정으로 실험 재현 가능
- ✅ **학습 시각화**: 학습 과정의 보상 변화를 그래프로 자동 저장
- ✅ **테스트 기능**: 학습된 모델을 불러와 시각적으로 확인

---

## 🗂️ 파일 구조

```
RL-CartPole-Project/
│
├── dqn_agent.py          # DQN 에이전트 및 신경망 구현
│   ├── QNetwork          # Q-Network (신경망 모델)
│   ├── ReplayBuffer      # 경험 재생 버퍼
│   └── DQNAgent          # DQN 에이전트 클래스
│
├── train.py              # 학습 메인 코드
│   ├── set_seed()        # 재현성을 위한 시드 고정
│   ├── plot_rewards()    # 학습 결과 그래프 생성
│   └── train_dqn()       # DQN 학습 메인 함수
│
├── test.py               # 학습된 모델 테스트
│   └── test_agent()      # 모델 로드 및 평가
│
├── requirements.txt      # 필요한 라이브러리 목록
├── README.md             # 프로젝트 설명 (현재 파일)
│
├── cartpole_dqn.pth      # 학습된 모델 가중치 (학습 후 생성)
└── result_graph.png      # 학습 결과 그래프 (학습 후 생성)
```

---

## 🔧 설치 방법

### 1. Python 버전 확인
이 프로젝트는 **Python 3.8 이상**에서 작동합니다.

```bash
python --version
```

### 2. 필요한 라이브러리 설치

프로젝트 디렉토리로 이동한 후, 다음 명령어로 필요한 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

설치되는 주요 라이브러리:
- `gymnasium[classic_control]`: 강화학습 환경 (CartPole-v1)
- `torch`: 딥러닝 프레임워크 (PyTorch)
- `numpy`: 수치 계산
- `matplotlib`: 데이터 시각화
- `tqdm`: 진행 상태 표시

---

## 🚀 실행 방법

### Google Colab에서 실행하기 (권장)

Google Colab에서 바로 실행할 수 있도록 준비된 노트북 파일을 사용하세요:

1. `CartPole_DQN_Colab.ipynb` 파일을 Google Colab에 업로드
2. 셀을 순서대로 실행
3. 한글 폰트가 자동으로 설정됩니다!

또는 Colab에서 직접 실행:
```python
# 1. 필요한 라이브러리 설치
!pip install -q gymnasium[classic_control] torch numpy matplotlib tqdm

# 2. 한글 폰트 설치
!apt-get install -y fonts-nanum > /dev/null 2>&1

# 3. 리포지토리 클론
!git clone https://github.com/taeryol/RL-CartPole-Project.git
%cd RL-CartPole-Project

# 4. 학습 실행
!python train.py
```

### 로컬 환경에서 실행하기

#### 1. 모델 학습 (`train.py`)

DQN 에이전트를 학습시키려면 다음 명령어를 실행합니다:

```bash
python train.py
```

**학습 과정:**
- 최대 1000 에피소드 동안 학습이 진행됩니다.
- 100 에피소드마다 평균 점수가 출력됩니다.
- 최근 100 에피소드의 평균 점수가 **195점 이상**이 되면 학습이 조기 종료됩니다.
- 학습이 완료되면 다음 파일들이 생성됩니다:
  - `cartpole_dqn.pth`: 학습된 모델 가중치
  - `result_graph.png`: 학습 과정의 보상 변화 그래프

**출력 예시:**
```
============================================================
CartPole-v1 DQN 학습 시작
============================================================
상태 공간 크기: 4
행동 공간 크기: 2
목표 평균 점수: 195.0
총 에피소드 수: 1000
============================================================

에피소드 100/1000 | 평균 점수 (최근 100): 45.23 | 현재 Epsilon: 0.605
에피소드 200/1000 | 평균 점수 (최근 100): 98.76 | 현재 Epsilon: 0.366
...
🎉 목표 달성! 에피소드 437에서 평균 점수 195.34 달성!
```

#### 2. 모델 테스트 (`test.py`)

학습된 모델을 불러와 시각적으로 확인하려면 다음 명령어를 실행합니다:

```bash
python test.py
```

**테스트 과정:**
- `cartpole_dqn.pth` 파일에서 학습된 모델을 불러옵니다.
- 5개의 에피소드를 실행하며 화면에 CartPole 시뮬레이션을 보여줍니다.
- 각 에피소드의 점수와 통계가 출력됩니다.

**출력 예시:**
```
============================================================
학습된 모델 테스트 시작 (5 에피소드)
============================================================
에피소드 1: 점수 = 500, 스텝 수 = 500
에피소드 2: 점수 = 500, 스텝 수 = 500
에피소드 3: 점수 = 500, 스텝 수 = 500
에피소드 4: 점수 = 498, 스텝 수 = 498
에피소드 5: 점수 = 500, 스텝 수 = 500

============================================================
테스트 결과 통계
============================================================
평균 점수: 499.60
최고 점수: 500
최저 점수: 498
표준 편차: 0.80
============================================================
✅ 우수! 평균 점수가 195점 이상입니다.
```

---

## 🧠 DQN 알고리즘 설명

### DQN (Deep Q-Network)란?

DQN은 Q-learning과 딥러닝을 결합한 알고리즘으로, 신경망을 사용하여 **행동 가치 함수(Q-function)**를 근사합니다.

### 주요 구성 요소

1. **Q-Network (행동 가치 함수 신경망)**
   - 상태를 입력받아 각 행동의 Q-값을 출력하는 신경망
   - 3개의 Fully Connected 레이어로 구성 (입력층 → 은닉층1 → 은닉층2 → 출력층)
   - 활성화 함수: ReLU

2. **Replay Buffer (경험 재생 버퍼)**
   - 에이전트가 환경에서 경험한 전이(transition)를 저장
   - 학습 시 무작위로 샘플링하여 데이터 간 상관관계 제거
   - 용량: 10,000개의 경험

3. **Target Network (타겟 네트워크)**
   - Q-Network와 동일한 구조를 가진 별도의 네트워크
   - 일정 주기마다 Q-Network의 가중치를 복사하여 업데이트
   - 학습 안정성 향상

4. **Epsilon-Greedy 정책**
   - 탐험(exploration)과 활용(exploitation)의 균형 유지
   - 초기에는 무작위 행동을 많이 선택하고, 학습이 진행될수록 학습된 정책을 따름
   - Epsilon 값이 1.0 → 0.01로 점진적으로 감소

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Learning Rate | 0.001 | 신경망 학습률 |
| Gamma (γ) | 0.99 | 할인율 (미래 보상의 중요도) |
| Epsilon Start | 1.0 | 초기 탐험 확률 |
| Epsilon End | 0.01 | 최소 탐험 확률 |
| Epsilon Decay | 0.995 | 에피소드마다 탐험 확률 감소율 |
| Buffer Size | 10,000 | Replay Buffer 크기 |
| Batch Size | 64 | 학습 시 미니배치 크기 |
| Target Update Frequency | 10 에피소드 | Target Network 업데이트 주기 |
| Hidden Size | 128 | 신경망 은닉층 뉴런 개수 |

---

## 📊 학습 결과

학습이 완료되면 `result_graph.png` 파일이 생성됩니다. 이 그래프는 다음 두 가지 정보를 보여줍니다:

1. **왼쪽 그래프**: 에피소드별 총 보상 변화
2. **오른쪽 그래프**: 100 에피소드 이동 평균 (목표 점수 195점 표시)

학습이 성공적으로 진행되면 이동 평균이 목표 점수인 195점을 넘어서게 됩니다.

---

## 📦 학습된 모델 파일 (`cartpole_dqn.pth`)

`cartpole_dqn.pth` 파일에는 다음 정보가 저장됩니다:

- Q-Network의 학습된 가중치
- Target Network의 가중치
- Optimizer의 상태
- 현재 Epsilon 값

이 파일을 사용하여 학습을 재개하거나, 학습된 에이전트를 테스트할 수 있습니다.

---

## 🎮 CartPole-v1 환경 정보

### 상태 공간 (State Space)
CartPole의 상태는 4차원 벡터로 표현됩니다:

1. **Cart Position** (카트 위치): -4.8 ~ 4.8
2. **Cart Velocity** (카트 속도): -∞ ~ +∞
3. **Pole Angle** (막대 각도): -0.418 ~ 0.418 라디안
4. **Pole Angular Velocity** (막대 각속도): -∞ ~ +∞

### 행동 공간 (Action Space)
2가지 행동이 가능합니다:

0. **왼쪽으로 밀기**
1. **오른쪽으로 밀기**

### 보상 (Reward)
- 매 스텝마다 +1의 보상을 받습니다.
- 막대가 넘어지거나 카트가 범위를 벗어나면 에피소드가 종료됩니다.

### 목표
- 에피소드당 최대 500 스텝을 유지하는 것이 목표입니다.
- 최근 100 에피소드의 평균 점수가 **195점 이상**이면 문제를 해결한 것으로 간주합니다.

---

## 🔍 코드 구조 상세 설명

### 1. `dqn_agent.py`

#### ReplayBuffer 클래스
```python
class ReplayBuffer:
    """경험 재생 버퍼"""
    def __init__(self, capacity)     # 버퍼 초기화
    def push(...)                    # 경험 저장
    def sample(batch_size)           # 무작위 샘플링
```

#### QNetwork 클래스
```python
class QNetwork(nn.Module):
    """Q-Network (신경망 모델)"""
    def __init__(self, state_size, action_size, hidden_size)
    def forward(self, state)         # 순전파
```

#### DQNAgent 클래스
```python
class DQNAgent:
    """DQN 에이전트"""
    def __init__(...)                # 초기화 (하이퍼파라미터 설정)
    def select_action(...)           # Epsilon-greedy 행동 선택
    def store_transition(...)        # 경험 저장
    def train(...)                   # Q-Network 학습
    def update_epsilon(...)          # Epsilon 감소
    def update_target_network(...)   # Target Network 업데이트
    def save(...)                    # 모델 저장
    def load(...)                    # 모델 불러오기
```

### 2. `train.py`

학습 메인 코드로 다음 기능을 수행합니다:

- Random Seed 고정 (`set_seed()`)
- 환경 및 에이전트 초기화
- 학습 루프 실행 (에피소드 반복)
- 진행 상황 출력 및 모델 저장
- 학습 결과 그래프 생성 (`plot_rewards()`)

### 3. `test.py`

테스트 코드로 다음 기능을 수행합니다:

- 저장된 모델 불러오기
- 렌더링 모드로 환경 실행
- 에이전트 성능 평가 및 통계 출력

---

## 🛠️ 추가 개선 아이디어

이 프로젝트를 더욱 발전시키고 싶다면 다음을 시도해보세요:

1. **Double DQN**: Target Network 선택 시 Q-Network를 사용하여 과대평가 문제 해결
2. **Dueling DQN**: 가치 함수와 이점 함수를 분리하여 학습
3. **Prioritized Experience Replay**: 중요한 경험에 더 높은 우선순위 부여
4. **하이퍼파라미터 튜닝**: 학습률, Epsilon 감소율, 배치 크기 등 조정
5. **다른 환경 적용**: LunarLander, Acrobot 등 다른 Gymnasium 환경에서 테스트

---

## 📚 참고 자료

- **논문**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- **Gymnasium 문서**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- **PyTorch 문서**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

---

## 👨‍💻 프로젝트 정보

- **환경**: CartPole-v1 (OpenAI Gymnasium)
- **알고리즘**: DQN (Deep Q-Network)
- **프레임워크**: PyTorch
- **목적**: 강화학습 학교 기말 프로젝트 제출용

---

## ❓ 문제 해결 (Troubleshooting)

### 1. 모델 파일을 찾을 수 없다는 오류
```
❌ 오류: 'cartpole_dqn.pth' 파일을 찾을 수 없습니다.
```
**해결 방법**: 먼저 `python train.py`를 실행하여 모델을 학습시키세요.

### 2. 학습이 너무 느림
**해결 방법**: GPU가 있는 경우 CUDA가 설치되어 있는지 확인하세요. PyTorch가 자동으로 GPU를 감지하여 사용합니다.

### 3. 학습이 수렴하지 않음
**해결 방법**: 
- `train.py`의 하이퍼파라미터를 조정해보세요.
- 학습률을 낮춰보세요 (예: 0.001 → 0.0005).
- Replay Buffer 크기를 늘려보세요.

### 4. 한글 폰트가 깨짐 (그래프)
**해결 방법**: 
- **Google Colab**: 코드가 자동으로 나눔고딕 폰트를 설치하고 설정합니다.
- **로컬 환경**: 시스템에 설치된 한글 폰트를 자동으로 감지하여 사용합니다.
- **수동 설정**: `train.py`의 `setup_korean_font()` 함수를 확인하세요.

---

## 📧 연락처

프로젝트 관련 문의사항은 GitHub Issues를 통해 남겨주세요.

---

**Happy Learning! 🚀**
