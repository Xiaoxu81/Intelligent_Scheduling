import torch
import numpy as np
from src.environment.gym_wrapper import SchedulingEnv
from src.models.drl_agent import PPOAgent

def main():
    # 1. 初始化环境和智能体
    num_strategies = 12
    env = SchedulingEnv(max_queue_size=10, num_strategies=num_strategies)
    agent = PPOAgent(max_queue_size=10, num_strategies=num_strategies)
    
    # 2. 强化学习在线训练
    print("Starting Strategy-Selection PPO Training...")
    print("Action space: Select from 12 strategy combinations (C01-C12)")
    num_episodes = 100
    update_timestep = 50
    timestep = 0
    memory = []
    
    strategy_usage = [0] * num_strategies  # 统计各策略使用频率
    
    for i_episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        episode_strategies = []
        
        while True:
            # DRL选择策略组合
            strategy_idx, log_prob, val = agent.select_action(obs)
            episode_strategies.append(strategy_idx)
            strategy_usage[strategy_idx] += 1
            
            # 执行选定的策略
            next_obs, reward, terminated, truncated, _ = env.step(strategy_idx)
            
            memory.append({
                'obs': obs,
                'action': strategy_idx,
                'log_prob': log_prob,
                'reward': reward,
                'value': val,
                'done': terminated
            })
            
            obs = next_obs
            episode_reward += reward
            timestep += 1
            
            # 定期更新 PPO 模型
            if timestep % update_timestep == 0 and len(memory) > 0:
                agent.update(memory)
                memory = []
                
            if terminated:
                break
        
        # 打印训练信息
        if i_episode % 10 == 0:
            strategy_id = f"C{episode_strategies[0] + 1:02d}" if episode_strategies else "N/A"
            print(f"Episode {i_episode:3d} | Reward: {episode_reward:7.2f} | "
                  f"First Strategy: {strategy_id} | Episode Steps: {len(episode_strategies)}")
    
    # 3. 打印策略使用统计
    print("\n" + "="*50)
    print("Strategy Usage Statistics:")
    print("="*50)
    for i in range(num_strategies):
        percentage = (strategy_usage[i] / sum(strategy_usage)) * 100 if sum(strategy_usage) > 0 else 0
        bar = "█" * int(percentage / 5)
        print(f"C{i+1:02d}: {strategy_usage[i]:4d} ({percentage:5.1f}%) {bar}")
    
    # 4. 保存模型
    torch.save(agent.model.state_dict(), "scheduling_agent.pth")
    print("\nTraining completed. Model saved to scheduling_agent.pth")
    
    # 5. 测试：展示智能体在不同状态下的策略选择
    print("\n" + "="*50)
    print("Testing Strategy Selection:")
    print("="*50)
    test_strategy_selection(agent, env)

def test_strategy_selection(agent, env, num_tests=5):
    """测试智能体在不同任务负载下的策略选择"""
    for i in range(num_tests):
        obs, _ = env.reset()
        probs = agent.get_strategy_probs(obs)
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        print(f"\nTest {i+1}:")
        print(f"  Ready tasks: {obs['global'][0]:.0f}")
        print(f"  Top 3 strategy preferences:")
        for rank, idx in enumerate(top3_idx, 1):
            strategy_id = f"C{idx + 1:02d}"
            print(f"    {rank}. {strategy_id}: {probs[idx]*100:.1f}%")

if __name__ == "__main__":
    main()
