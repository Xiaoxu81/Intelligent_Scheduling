from src.environment.gym_wrapper import SchedulingEnv

def test_gym_env():
    env = SchedulingEnv(max_queue_size=10)
    obs, _ = env.reset()
    
    print("\n--- Testing Gymnasium Wrapper ---")
    print(f"Initial Observation (Global): {obs['global']}")
    print(f"First 2 tasks in observation:\n{obs['tasks'][:2]}")
    
    # 执行一个动作：调度就绪队列中的第一个任务 (Action 0)
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"\nStep 1 (Action 0) -> Reward: {reward:.2f}, Terminated: {terminated}")
    
    # 连续执行动作直到结束
    steps = 0
    while not terminated and steps < 50:
        # 简单的贪婪策略：始终尝试调度第一个就绪任务，如果没有则选择等待（Action 10）
        ready_count = int(obs['global'][0])
        idle_count = int(obs['global'][1])
        
        action = 0 if ready_count > 0 and idle_count > 0 else 10
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
    print(f"\nSimulation finished in {steps} steps.")
    print(f"Final Global State: {obs['global']}")
    assert terminated == True
    print("--- Gymnasium Wrapper Test Passed ---")

if __name__ == "__main__":
    test_gym_env()
