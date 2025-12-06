"""
CartPole-v1 환경에서 DQN 에이전트 학습
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from dqn_agent import DQNAgent
from tqdm import tqdm
import platform
import os


def setup_korean_font():
    """
    한글 폰트 설정 (Google Colab 및 다양한 환경 지원)
    """
    import matplotlib.font_manager as fm
    
    # 시스템 종류 확인
    system = platform.system()
    
    # Google Colab 환경인 경우 폰트 설치
    try:
        if 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython()):
            print("Google Colab 환경 감지 - 한글 폰트 설치 중...")
            import subprocess
            subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
            
            # 폰트 캐시 삭제 및 재생성
            import shutil
            font_cache_dir = os.path.expanduser('~/.cache/matplotlib')
            if os.path.exists(font_cache_dir):
                shutil.rmtree(font_cache_dir)
            
            # 나눔고딕 폰트 사용
            plt.rcParams['font.family'] = 'NanumGothic'
            print("한글 폰트 설정 완료!")
            return
    except:
        pass
    
    # 사용 가능한 한글 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 우선순위대로 한글 폰트 찾기
    korean_fonts = [
        'NanumGothic',           # Google Colab, Ubuntu
        'Malgun Gothic',         # Windows
        'AppleGothic',           # macOS
        'Noto Sans CJK KR',      # Linux
        'DejaVu Sans'            # 기본 폰트 (한글 미지원)
    ]
    
    selected_font = None
    for font in korean_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = selected_font
        print(f"한글 폰트 설정: {selected_font}")
    else:
        # 한글 폰트를 찾지 못한 경우 영문으로 대체
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("한글 폰트를 찾을 수 없어 영문으로 표시됩니다.")
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    """
    재현성을 위한 Random Seed 고정
    
    Args:
        seed (int): 난수 시드 값 (기본값: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_rewards(rewards, save_path='result_graph.png'):
    """
    학습 과정의 보상 변화를 그래프로 그리고 저장
    
    Args:
        rewards (list): 에피소드별 총 보상 리스트
        save_path (str): 그래프를 저장할 파일 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 에피소드별 보상 그래프
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('DQN Training Progress - Episode Rewards', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 이동 평균 (100 에피소드 단위)
    plt.subplot(1, 2, 2)
    if len(rewards) >= 100:
        moving_avg = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
        plt.plot(moving_avg, color='red', linewidth=2, label='Moving Average (100 episodes)')
        plt.axhline(y=195, color='green', linestyle='--', label='Target Score (195)')
    else:
        plt.plot(rewards, color='red', linewidth=2, label='Total Reward')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('DQN Training Progress - Moving Average', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프가 '{save_path}'에 저장되었습니다.")
    plt.close()


def train_dqn(
    n_episodes=1000,
    max_steps=500,
    target_score=195.0,
    print_interval=100,
    model_save_path='cartpole_dqn.pth'
):
    """
    DQN 에이전트를 CartPole-v1 환경에서 학습
    
    Args:
        n_episodes (int): 총 학습 에피소드 수 (기본값: 1000)
        max_steps (int): 에피소드당 최대 스텝 수 (기본값: 500)
        target_score (float): 목표 평균 점수 (기본값: 195.0)
        print_interval (int): 진행 상황 출력 간격 (기본값: 100 에피소드)
        model_save_path (str): 모델 저장 경로 (기본값: 'cartpole_dqn.pth')
    """
    
    # 한글 폰트 설정 (Google Colab 지원)
    setup_korean_font()
    
    # Random Seed 고정 (재현성 확보)
    set_seed(42)
    
    # CartPole-v1 환경 생성
    env = gym.make('CartPole-v1')
    
    # 상태 및 행동 공간 크기 확인
    state_size = env.observation_space.shape[0]  # CartPole: 4 (위치, 속도, 각도, 각속도)
    action_size = env.action_space.n  # CartPole: 2 (왼쪽, 오른쪽)
    
    print("=" * 60)
    print("CartPole-v1 DQN 학습 시작")
    print("=" * 60)
    print(f"상태 공간 크기: {state_size}")
    print(f"행동 공간 크기: {action_size}")
    print(f"목표 평균 점수: {target_score}")
    print(f"총 에피소드 수: {n_episodes}")
    print("=" * 60)
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,      # 학습률
        gamma=0.99,               # 할인율
        epsilon_start=1.0,        # 초기 탐험 확률
        epsilon_end=0.01,         # 최소 탐험 확률
        epsilon_decay=0.995,      # 탐험 확률 감소율
        buffer_size=10000,        # Replay Buffer 크기
        batch_size=64,            # 미니배치 크기
        target_update_freq=10     # Target Network 업데이트 빈도 (에피소드 단위)
    )
    
    # 학습 기록 저장 리스트
    episode_rewards = []  # 에피소드별 총 보상
    recent_scores = []    # 최근 100개 에피소드의 점수 (목표 달성 확인용)
    
    # 학습 루프
    for episode in tqdm(range(1, n_episodes + 1), desc="학습 진행"):
        # 환경 초기화
        state, _ = env.reset(seed=42 + episode)
        total_reward = 0
        
        # 에피소드 실행
        for step in range(max_steps):
            # 행동 선택 (epsilon-greedy)
            action = agent.select_action(state, training=True)
            
            # 환경에서 행동 실행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 경험을 Replay Buffer에 저장
            agent.store_transition(state, action, reward, next_state, done)
            
            # 에이전트 학습
            loss = agent.train()
            
            # 상태 업데이트
            state = next_state
            total_reward += reward
            
            # 에피소드 종료
            if done:
                break
        
        # 에피소드 종료 후 처리
        episode_rewards.append(total_reward)
        recent_scores.append(total_reward)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        # Epsilon 감소 (탐험 확률 줄이기)
        agent.update_epsilon()
        
        # Target Network 업데이트 (일정 주기마다)
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # 진행 상황 출력 (100 에피소드마다)
        if episode % print_interval == 0:
            avg_score = np.mean(recent_scores)
            print(f"\n에피소드 {episode}/{n_episodes} | "
                  f"평균 점수 (최근 100): {avg_score:.2f} | "
                  f"현재 Epsilon: {agent.epsilon:.3f}")
        
        # 목표 점수 달성 확인 (최근 100개 에피소드 평균)
        if len(recent_scores) >= 100:
            avg_score = np.mean(recent_scores)
            if avg_score >= target_score:
                print("\n" + "=" * 60)
                print(f"🎉 목표 달성! 에피소드 {episode}에서 평균 점수 {avg_score:.2f} 달성!")
                print("=" * 60)
                agent.save(model_save_path)
                break
    
    # 학습 종료 처리
    else:
        # 최대 에피소드 도달 시 모델 저장
        print("\n" + "=" * 60)
        print(f"학습 완료! 최종 평균 점수: {np.mean(recent_scores):.2f}")
        print("=" * 60)
        agent.save(model_save_path)
    
    # 환경 종료
    env.close()
    
    # 학습 결과 그래프 저장
    plot_rewards(episode_rewards)
    
    # 최종 통계 출력
    print("\n" + "=" * 60)
    print("학습 통계")
    print("=" * 60)
    print(f"총 에피소드: {len(episode_rewards)}")
    print(f"최고 점수: {max(episode_rewards):.2f}")
    print(f"평균 점수: {np.mean(episode_rewards):.2f}")
    print(f"최종 100 에피소드 평균: {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    # 학습 실행
    train_dqn(
        n_episodes=1000,           # 최대 에피소드 수
        max_steps=500,             # 에피소드당 최대 스텝
        target_score=195.0,        # 목표 평균 점수
        print_interval=100,        # 진행 상황 출력 간격
        model_save_path='cartpole_dqn.pth'  # 모델 저장 경로
    )
