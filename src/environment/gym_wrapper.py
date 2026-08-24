import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.models.task import Task, TaskStatus
from src.models.resource import Resource, ResourceStatus
from src.environment.simulation import SimulationEnv
from src.strategies.heuristics import get_scheduler_by_id

class SchedulingEnv(gym.Env):
    """
    策略选择型调度强化学习环境
    动作空间：选择12种策略组合之一 (C01-C12)
    """
    def __init__(self, max_queue_size: int = 10, num_strategies: int = 12):
        super(SchedulingEnv, self).__init__()
        self.max_queue_size = max_queue_size
        self.num_strategies = num_strategies
        self.sim = SimulationEnv()
        
        # 定义观察空间
        task_feat_dim = 7
        self.observation_space = spaces.Dict({
            "tasks": spaces.Box(low=-1, high=1000, shape=(max_queue_size, task_feat_dim), dtype=np.float32),
            "global": spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)
        })
        
        # 定义动作空间: 选择12种策略组合之一
        self.action_space = spaces.Discrete(num_strategies)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.sim = SimulationEnv()
        # 初始化一些测试任务
        self._seed_tasks()
        
        obs = self._get_obs()
        return obs, {}

    def _seed_tasks(self):
        """初始化基础任务负载"""
        for i in range(5):
            t = Task(f"T{i}", priority=np.random.randint(1, 10), duration=np.random.uniform(5, 15), arrival_time=0)
            if np.random.random() < 0.2:
                t.is_realtime = True
                t.deadline = 20.0
            self.sim.add_task(t)
        
        self.sim.add_resource(Resource("R1", "Machine"))
        self.sim.add_resource(Resource("R2", "Machine"))
        
        # 推进到第一个事件
        self.sim.step()

    def _get_obs(self):
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]
        # 按到达时间排序取前 K 个
        ready_tasks.sort(key=lambda x: x.arrival_time)
        
        task_feats = np.zeros((self.max_queue_size, 7), dtype=np.float32) - 1.0
        for i, t in enumerate(ready_tasks[:self.max_queue_size]):
            wait_time = self.sim.current_time - (t.wait_start_time if t.wait_start_time else 0)
            task_feats[i] = [
                t.priority,
                t.duration,
                t.arrival_time,
                t.deadline if t.deadline else -1,
                t.remaining_time,
                1.0 if getattr(t, 'is_realtime', False) else 0.0,
                wait_time
            ]
            
        idle_res_count = len([r for r in self.sim.resources.values() if r.status == ResourceStatus.IDLE])
        global_feats = np.array([len(ready_tasks), idle_res_count], dtype=np.float32)
        
        return {"tasks": task_feats, "global": global_feats}

    def step(self, action: int):
        """
        执行策略选择动作
        action: 0-11 对应 C01-C12 策略组合
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. 根据动作选择对应的调度策略
        strategy_id = f"C{action + 1:02d}"
        scheduler = get_scheduler_by_id(strategy_id)
        self.sim.scheduler = scheduler
        
        # 2. 使用选定的策略执行调度
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]
        resources = list(self.sim.resources.values())
        
        # 执行调度决策
        decisions = scheduler.schedule(ready_tasks, resources, self.sim.current_time)
        
        # 3. 执行调度结果
        scheduled_count = 0
        for task, resource in decisions:
            if resource.status == ResourceStatus.IDLE and task.status == TaskStatus.READY:
                self.sim._execute_task(task, resource)
                scheduled_count += 1
        
        # 根据调度效果给予奖励
        if scheduled_count > 0:
            reward += 1.0 * scheduled_count
        else:
            reward -= 0.2  # 策略未能调度任何任务的轻微惩罚

        # 4. 推进仿真引擎
        if self.sim.event_queue:
            prev_time = self.sim.current_time
            event = self.sim.step()
            dt = self.sim.current_time - prev_time
            
            # 时间惩罚
            reward -= 0.1 * len(ready_tasks) * dt
            
            # 检查任务超时
            for t in self.sim.tasks.values():
                if t.status != TaskStatus.COMPLETED and t.deadline and self.sim.current_time > t.deadline:
                    reward -= 10.0
        else:
            # 所有任务完成
            if all(t.status == TaskStatus.COMPLETED for t in self.sim.tasks.values()):
                terminated = True
                reward += 20.0

        return self._get_obs(), reward, terminated, truncated, {}
