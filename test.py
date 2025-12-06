"""
학습된 DQN 모델 테스트 및 시각화
저장된 모델을 불러와 CartPole-v1 환경에서 성능을 확인합니다.
"""

import gymnasium as gym
import numpy as np
import time
from dqn_agent import DQNAgent


def test_agent(
    model_path='cartpole_dqn.pth',
    n_episodes=5,
    render=True,
    sleep_time=0.02
):
    """
    학습된 DQN 에이전트를 테스트
    
    Args:
        model_path (str): 불러올 모델 파일 경로 (기본값: 'cartpole_dqn.pth')
        n_episodes (int): 테스트할 에피소드 수 (기본값: 5)
        render (bool): 화면 렌더링 여부 (기본값: True)
        sleep_time (float): 렌더링 시 스텝 간 대기 시간 (기본값: 0.02초)
    """
    
    # 렌더링 모드로 환경 생성
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
        print("=" * 60)
        print("화면 렌더링 모드로 테스트를 시작합니다.")
        print("창을 닫으려면 프로그램을 종료하세요.")
        print("=" * 60)
    else:
        env = gym.make('CartPole-v1')
        print("=" * 60)
        print("렌더링 없이 테스트를 시작합니다.")
        print("=" * 60)
    
    # 상태 및 행동 공간 크기
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # DQN 에이전트 생성 및 모델 불러오기
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"❌ 오류: '{model_path}' 파일을 찾을 수 없습니다.")
        print("먼저 'python train.py'를 실행하여 모델을 학습시키세요.")
        env.close()
        return
    
    # 테스트 결과 저장 리스트
    test_scores = []
    
    print("\n" + "=" * 60)
    print(f"학습된 모델 테스트 시작 ({n_episodes} 에피소드)")
    print("=" * 60)
    
    # 테스트 루프
    for episode in range(1, n_episodes + 1):
        # 환경 초기화
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        # 에피소드 실행
        while True:
            # 학습된 정책으로 행동 선택 (탐험 없음)
            action = agent.select_action(state, training=False)
            
            # 환경에서 행동 실행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 상태 업데이트
            state = next_state
            total_reward += reward
            steps += 1
            
            # 렌더링 시 약간의 딜레이 추가 (시각적 확인을 위해)
            if render:
                time.sleep(sleep_time)
            
            # 에피소드 종료
            if done:
                break
        
        # 에피소드 결과 저장 및 출력
        test_scores.append(total_reward)
        print(f"에피소드 {episode}: 점수 = {total_reward:.0f}, 스텝 수 = {steps}")
    
    # 환경 종료
    env.close()
    
    # 테스트 통계 출력
    print("\n" + "=" * 60)
    print("테스트 결과 통계")
    print("=" * 60)
    print(f"평균 점수: {np.mean(test_scores):.2f}")
    print(f"최고 점수: {max(test_scores):.0f}")
    print(f"최저 점수: {min(test_scores):.0f}")
    print(f"표준 편차: {np.std(test_scores):.2f}")
    print("=" * 60)
    
    # 성능 평가
    avg_score = np.mean(test_scores)
    if avg_score >= 195:
        print("✅ 우수! 평균 점수가 195점 이상입니다.")
    elif avg_score >= 100:
        print("⚠️  양호. 평균 점수가 100점 이상이지만 개선의 여지가 있습니다.")
    else:
        print("❌ 부족. 추가 학습이 필요합니다.")
    print("=" * 60)


if __name__ == "__main__":
    # 테스트 실행
    # render=True: 화면에 CartPole 시뮬레이션을 보여줌
    # render=False: 렌더링 없이 빠르게 테스트
    test_agent(
        model_path='cartpole_dqn.pth',  # 불러올 모델 파일
        n_episodes=5,                    # 테스트 에피소드 수
        render=False,                    # 화면 렌더링 여부
        sleep_time=0.02                  # 렌더링 시 프레임 간 대기 시간
    )
