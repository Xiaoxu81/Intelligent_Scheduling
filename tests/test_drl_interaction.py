from src.environment.gym_wrapper import SchedulingEnv
from src.models.drl_agent import PPOAgent
import torch

def test_drl_interaction():
    env = SchedulingEnv(max_queue_size=10)
    agent = PPOAgent(max_queue_size=10)
    
    obs, _ = env.reset()
    
    print("\n--- Testing DRL Agent Interaction ---")
    
    steps = 0
    total_reward = 0
    while steps < 20:
        # Agent 根据观察选择动作
        action, log_prob, _ = agent.select_action(obs)
        
        # 环境执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if steps % 5 == 0:
            print(f"Step {steps}: Action chosen = {action}, Current Reward = {reward:.2f}, Total = {total_reward:.2f}")
            
        if terminated:
            print("Episode terminated!")
            break
            
    print(f"\nInteraction test finished. Total steps: {steps}, Final Reward: {total_reward:.2f}")
    print("--- DRL Agent Interaction Test Passed ---")

if __name__ == "__main__":
    test_drl_interaction()
