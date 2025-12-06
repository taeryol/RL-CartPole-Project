"""
DQN (Deep Q-Network) 에이전트 구현
CartPole-v1 환경을 위한 강화학습 에이전트
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple


# 경험 저장을 위한 구조체 정의
# Transition: (상태, 행동, 보상, 다음 상태, 종료 여부)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    경험 재생 버퍼 (Replay Buffer)
    
    에이전트가 경험한 전이(transition)를 저장하고,
    학습 시 무작위로 샘플링하여 제공합니다.
    이를 통해 데이터 간 상관관계를 줄이고 학습 안정성을 높입니다.
    """
    
    def __init__(self, capacity):
        """
        Args:
            capacity (int): 버퍼의 최대 크기 (최대 저장 가능한 경험의 수)
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        경험을 버퍼에 저장
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        버퍼에서 무작위로 배치 샘플링
        
        Args:
            batch_size (int): 샘플링할 경험의 개수
            
        Returns:
            list: 샘플링된 Transition 객체들의 리스트
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """버퍼에 저장된 경험의 개수 반환"""
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-Network (행동 가치 함수 근사 신경망)
    
    상태를 입력받아 각 행동에 대한 Q-값을 출력하는 신경망입니다.
    3개의 fully connected 레이어로 구성되어 있습니다.
    """
    
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Args:
            state_size (int): 상태 공간의 차원 (CartPole: 4)
            action_size (int): 행동 공간의 차원 (CartPole: 2)
            hidden_size (int): 은닉층의 뉴런 개수 (기본값: 128)
        """
        super(QNetwork, self).__init__()
        
        # 첫 번째 fully connected 레이어: 상태 -> 은닉층1
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # 두 번째 fully connected 레이어: 은닉층1 -> 은닉층2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 세 번째 fully connected 레이어: 은닉층2 -> Q-값 출력
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """
        순전파 (Forward pass)
        
        Args:
            state (torch.Tensor): 입력 상태
            
        Returns:
            torch.Tensor: 각 행동에 대한 Q-값
        """
        # ReLU 활성화 함수를 사용한 순차적 레이어 통과
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # 출력층은 활성화 함수 없음
        
        return q_values


class DQNAgent:
    """
    DQN 에이전트
    
    Q-Network를 사용하여 최적의 정책을 학습하는 에이전트입니다.
    Target Network를 사용하여 학습 안정성을 높입니다.
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    ):
        """
        Args:
            state_size (int): 상태 공간의 차원
            action_size (int): 행동 공간의 차원
            learning_rate (float): 학습률 (기본값: 0.001)
            gamma (float): 할인율 (미래 보상의 중요도, 기본값: 0.99)
            epsilon_start (float): 초기 탐험 확률 (기본값: 1.0)
            epsilon_end (float): 최소 탐험 확률 (기본값: 0.01)
            epsilon_decay (float): 탐험 확률 감소율 (기본값: 0.995)
            buffer_size (int): Replay Buffer의 크기 (기본값: 10000)
            batch_size (int): 학습 시 배치 크기 (기본값: 64)
            target_update_freq (int): Target Network 업데이트 빈도 (기본값: 10 에피소드)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon_start  # 현재 탐험 확률 (epsilon-greedy)
        self.epsilon_end = epsilon_end  # 최소 탐험 확률
        self.epsilon_decay = epsilon_decay  # 탐험 확률 감소율
        self.batch_size = batch_size  # 미니배치 크기
        self.target_update_freq = target_update_freq  # Target Network 업데이트 주기
        
        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network (학습용 네트워크)
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        
        # Target Network (안정적인 학습을 위한 타겟 네트워크)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target Network는 추론 모드로 설정
        
        # 옵티마이저 (Adam 사용)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay Buffer 초기화
        self.memory = ReplayBuffer(buffer_size)
        
        # 학습 진행 추적 변수
        self.steps = 0
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy 정책에 따라 행동 선택
        
        Args:
            state (np.ndarray): 현재 상태
            training (bool): 학습 모드 여부 (False일 경우 탐험 없이 최적 행동만 선택)
            
        Returns:
            int: 선택된 행동
        """
        # 학습 중이고 탐험을 할 경우 (epsilon 확률로 무작위 행동)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # 활용 (exploitation): Q-값이 가장 높은 행동 선택
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        경험을 Replay Buffer에 저장
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Replay Buffer에서 샘플링하여 Q-Network 학습
        
        Returns:
            float: 학습 손실값 (loss), 버퍼가 부족하면 None
        """
        # 버퍼에 충분한 경험이 쌓이지 않았으면 학습하지 않음
        if len(self.memory) < self.batch_size:
            return None
        
        # 미니배치 샘플링
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # 배치 데이터를 텐서로 변환
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # 현재 Q-값 계산: Q(s, a)
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # 다음 상태의 최대 Q-값 계산: max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].unsqueeze(1)
            # 타겟 Q-값 계산: r + gamma * max_a' Q_target(s', a') * (1 - done)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # 손실 함수 계산 (Huber Loss / Smooth L1 Loss)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 (학습 안정성 향상)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_epsilon(self):
        """
        Epsilon 값을 감소시켜 탐험 확률을 줄임
        학습이 진행될수록 탐험보다 활용을 선호하도록 함
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """
        Target Network를 Q-Network의 가중치로 업데이트
        일정 주기마다 호출하여 학습 안정성을 높임
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """
        학습된 모델 저장
        
        Args:
            filepath (str): 모델을 저장할 파일 경로
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load(self, filepath):
        """
        저장된 모델 불러오기
        
        Args:
            filepath (str): 불러올 모델 파일 경로
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"모델이 {filepath}에서 불러와졌습니다.")
