import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.models.task import Task, TaskStatus
from src.models.task_demand import DEMAND_KEYS
from src.models.resource import Resource, ResourceStatus
from src.environment.simulation import SimulationEnv
from src.strategies.heuristics import get_scheduler_by_id

class SchedulingEnv(gym.Env):
    """
    策略选择型调度强化学习环境
    动作空间：选择12种策略组合之一 (C01-C12)
    """
    def __init__(
        self,
        max_queue_size: int = 10,
        num_strategies: int = 12,
        max_resource_size: int = 4,
        max_assignments_per_step: Optional[int] = None,
    ):
        super(SchedulingEnv, self).__init__()
        self.max_queue_size = max_queue_size
        self.num_strategies = num_strategies
        self.strategy_ids = [f"C{i + 1:02d}" for i in range(num_strategies)]
        self.max_resource_size = max_resource_size
        if max_assignments_per_step is not None and max_assignments_per_step < 1:
            raise ValueError("max_assignments_per_step must be positive or None")
        self.max_assignments_per_step = max_assignments_per_step
        self.sim = SimulationEnv()
        
        # 报告中的 S、T、R、wT，保留 global 作为旧代码兼容别名。
        self.observation_space = spaces.Dict({
            "system": spaces.Box(low=-1, high=1000, shape=(6,), dtype=np.float32),
            "tasks": spaces.Box(low=-1, high=1000, shape=(max_queue_size, 9), dtype=np.float32),
            "demands": spaces.Box(low=-1, high=1, shape=(max_queue_size, len(DEMAND_KEYS)), dtype=np.float32),
            "resources": spaces.Box(low=-1, high=1000, shape=(max_resource_size, 5), dtype=np.float32),
            "weights": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            "global": spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32),
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
        rng = self.np_random
        for i in range(5):
            t = Task(
                f"T{i}",
                priority=int(rng.integers(1, 10)),
                duration=float(rng.uniform(5, 15)),
                arrival_time=0,
                capability_requirements={"machine": float(rng.integers(1, 3))},
            )
            if float(rng.random()) < 0.2:
                t.is_realtime = True
                t.deadline = 20.0
            self.sim.add_task(t)
        
        for i in range(min(2, self.max_resource_size)):
            self.sim.add_resource(
                Resource(
                    f"R{i + 1}",
                    "Machine",
                    capacity=1.0,
                    capabilities={"machine": 2.0},
                    reliability=1.0,
                )
            )
        
        # 推进到第一个事件
        self.sim.step()

    def _get_obs(self):
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]
        # 按到达时间排序取前 K 个
        ready_tasks.sort(key=lambda x: x.arrival_time)
        
        task_feats = np.zeros((self.max_queue_size, 9), dtype=np.float32) - 1.0
        demand_feats = np.zeros((self.max_queue_size, len(DEMAND_KEYS)), dtype=np.float32) - 1.0
        first_weights = np.full(4, 0.25, dtype=np.float32)
        downstream_counts = {task.task_id: 0 for task in self.sim.tasks.values()}
        for task in self.sim.tasks.values():
            for dependency in task.dependencies:
                if dependency in downstream_counts:
                    downstream_counts[dependency] += 1
        total_resources = len(self.sim.resources)
        for i, t in enumerate(ready_tasks[:self.max_queue_size]):
            wait_time = self.sim.current_time - (t.wait_start_time if t.wait_start_time else 0)
            first_weights = np.array([
                t.demand_features(
                    current_time=self.sim.current_time,
                    downstream_count=downstream_counts.get(t.task_id, 0),
                    feasible_resource_count=sum(
                        1 for resource in self.sim.resources.values()
                        if all(resource.capabilities.get(key, 0.0) >= value for key, value in t.capability_requirements.items())
                    ),
                    total_resource_count=total_resources,
                )["weights"][key]
                for key in ("time", "throughput", "cost", "stability")
            ], dtype=np.float32)
            demand = t.demand_features(
                current_time=self.sim.current_time,
                downstream_count=downstream_counts.get(t.task_id, 0),
                feasible_resource_count=sum(
                    1 for resource in self.sim.resources.values()
                    if all(resource.capabilities.get(key, 0.0) >= value for key, value in t.capability_requirements.items())
                ),
                total_resource_count=total_resources,
            )
            demand_feats[i] = [demand[key] for key in DEMAND_KEYS]
            task_feats[i] = [
                t.priority,
                t.duration,
                t.arrival_time,
                t.deadline if t.deadline else -1,
                t.remaining_time,
                1.0 if getattr(t, 'is_realtime', False) else 0.0,
                wait_time,
                t.capability_requirements.get("machine", 0.0),
                float(len(t.dependencies)),
            ]
            
        idle_res_count = len([r for r in self.sim.resources.values() if r.status == ResourceStatus.IDLE])
        running_count = len([r for r in self.sim.resources.values() if r.status == ResourceStatus.BUSY])
        pending_count = len([t for t in self.sim.tasks.values() if t.status == TaskStatus.PENDING])
        failed_count = len([t for t in self.sim.tasks.values() if t.status == TaskStatus.FAILED])
        pending_workload = sum(
            t.remaining_time
            for t in self.sim.tasks.values()
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        )
        system_feats = np.array([
            len(ready_tasks),
            idle_res_count,
            running_count,
            pending_count,
            failed_count,
            pending_workload,
        ], dtype=np.float32)

        resource_feats = np.zeros((self.max_resource_size, 5), dtype=np.float32) - 1.0
        for i, resource in enumerate(list(self.sim.resources.values())[:self.max_resource_size]):
            resource_feats[i] = [
                resource.capacity,
                resource.capabilities.get("machine", 0.0),
                resource.reliability,
                1.0 if resource.status == ResourceStatus.BUSY else 0.0,
                1.0 if resource.status == ResourceStatus.FAULT else 0.0,
            ]

        global_feats = np.array([len(ready_tasks), idle_res_count], dtype=np.float32)
        return {
            "system": system_feats,
            "tasks": task_feats,
            "demands": demand_feats,
            "resources": resource_feats,
            "weights": first_weights,
            "global": global_feats,
        }

    def step(self, action: int):
        """
        执行策略选择动作
        action: 0-11 对应 C01-C12 策略组合
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. 根据动作选择对应的调度策略
        strategy_id = self.strategy_ids[int(action)]
        scheduler = get_scheduler_by_id(strategy_id)
        self.sim.scheduler = scheduler
        
        # 2. 使用选定的策略执行调度
        ready_tasks = [t for t in self.sim.tasks.values() if t.status == TaskStatus.READY]
        resources = list(self.sim.resources.values())
        
        # 执行调度决策
        decisions = scheduler.schedule(ready_tasks, resources, self.sim.current_time)
        
        # 3. 执行调度结果
        scheduled_count = 0
        if self.max_assignments_per_step is not None:
            decisions = decisions[: self.max_assignments_per_step]
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
            if all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in self.sim.tasks.values()):
                terminated = True
                if all(t.status == TaskStatus.COMPLETED for t in self.sim.tasks.values()):
                    completion_times = [
                        max(0.0, t.end_time - t.arrival_time)
                        for t in self.sim.tasks.values()
                        if t.end_time is not None
                    ]
                    if completion_times:
                        reward -= sum(completion_times) / len(completion_times)
                    reward += 20.0
        else:
            # 所有任务完成
            if all(t.status == TaskStatus.COMPLETED for t in self.sim.tasks.values()):
                terminated = True
                completion_times = [
                    max(0.0, t.end_time - t.arrival_time)
                    for t in self.sim.tasks.values()
                    if t.end_time is not None
                ]
                if completion_times:
                    reward -= sum(completion_times) / len(completion_times)
                reward += 20.0

        return self._get_obs(), reward, terminated, truncated, {}
