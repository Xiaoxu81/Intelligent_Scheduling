import numpy as np
from src.models.task import TaskStatus
from src.environment.gym_wrapper import SchedulingEnv
from src.strategies.heuristics import get_scheduler_by_id

class ExpertCollector:
    def __init__(self, env: SchedulingEnv, combination_id: str = "C05"):
        self.env = env
        self.expert = get_scheduler_by_id(combination_id)

    def collect_data(self, num_episodes=10):
        expert_data = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            terminated = False
            while not terminated:
                # 获取环境内部状态以供专家决策
                ready_tasks = [t for t in self.env.sim.tasks.values() if t.status == TaskStatus.READY]
                resources = list(self.env.sim.resources.values())
                
                # 专家决策
                decisions = self.expert.schedule(ready_tasks, resources, self.env.sim.current_time)
                
                if decisions:
                    # 专家选择的任务在 ready_tasks 中的索引（对应动作空间）
                    target_task = decisions[0][0]
                    # 在 obs['tasks'] 中找到对应的索引
                    action = -1
                    # ready_tasks 在 env.step 中是按到达时间排序的
                    ready_tasks_sorted = sorted(ready_tasks, key=lambda x: x.arrival_time)
                    for i, t in enumerate(ready_tasks_sorted[:self.env.max_queue_size]):
                        if t.task_id == target_task.task_id:
                            action = i
                            break
                    
                    if action == -1: # 如果任务不在前 K 个，或者专家选了不调度的策略
                        action = self.env.max_queue_size
                else:
                    action = self.env.max_queue_size
                
                # 记录数据
                expert_data.append((obs, action))
                
                # 执行动作推进一步
                obs, reward, terminated, truncated, _ = self.env.step(action)
                
        return expert_data
