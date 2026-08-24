from typing import List, Tuple, Dict
from src.strategies.base import BaseScheduler, MainMode, OrderingStrategy
from src.models.task import Task, TaskStatus
from src.models.resource import Resource, ResourceStatus

class UnifiedScheduler(BaseScheduler):
    """
    统一调度器：实现 12 种策略组合 (C01-C12)
    """
    def __init__(self, mode: MainMode, ordering: OrderingStrategy):
        super().__init__(mode, ordering)

    def _get_ordering_score(self, task: Task, current_time: float) -> float:
        """计算排序分数，分数越高优先级越高"""
        
        # 1. 基础排序策略逻辑
        if self.ordering == OrderingStrategy.O1_GLOBAL_PRIORITY:
            score = task.priority
        elif self.ordering == OrderingStrategy.O2_RESPONSE_RATIO:
            wait_time = current_time - (task.wait_start_time if task.wait_start_time is not None else current_time)
            # 响应比 = (等待时间 + 服务时间) / 服务时间
            score = (wait_time + task.duration) / task.duration if task.duration > 0 else wait_time
        elif self.ordering == OrderingStrategy.O3_DEADLINE:
            # 截止时间越早，分数越高
            score = -task.deadline if task.deadline is not None else -float('inf')
        elif self.ordering == OrderingStrategy.O4_FIFO:
            # 到达时间越早，分数越高
            score = -task.arrival_time
        else:
            score = 0.0

        # 2. 主模式动态修正 (M3: 动态权重)
        if self.mode == MainMode.M3_DYNAMIC_WEIGHT:
            # 模拟动态负载适配：根据等待时间动态增加权重（防止长任务饥饿）
            aging_factor = (current_time - task.arrival_time) * 0.1
            score += aging_factor

        return score

    def schedule(self, ready_tasks: List[Task], resources: List[Resource], current_time: float) -> List[Tuple[Task, Resource]]:
        if not ready_tasks:
            return []

        decisions = []
        idle_resources = [r for r in resources if r.status == ResourceStatus.IDLE]
        
        if not idle_resources:
            return []

        # 根据主模式 M2 处理逻辑：实时任务优先
        if self.mode == MainMode.M2_REALTIME_PREEMPT:
            rt_tasks = [t for t in ready_tasks if getattr(t, 'is_realtime', False)]
            non_rt_tasks = [t for t in ready_tasks if not getattr(t, 'is_realtime', False)]
            
            # 实时任务按排序策略执行
            rt_tasks.sort(key=lambda t: self._get_ordering_score(t, current_time), reverse=True)
            non_rt_tasks.sort(key=lambda t: self._get_ordering_score(t, current_time), reverse=True)
            
            sorted_tasks = rt_tasks + non_rt_tasks
        else:
            # M1 和 M3 统一按分数排序
            sorted_tasks = sorted(ready_tasks, key=lambda t: self._get_ordering_score(t, current_time), reverse=True)

        for task in sorted_tasks:
            if not idle_resources:
                break
            res = idle_resources.pop(0)
            decisions.append((task, res))
            
        return decisions

def get_scheduler_by_id(combination_id: str) -> UnifiedScheduler:
    """根据组合 ID (C01-C12) 获取对应的调度器实例"""
    mapping = {
        "C01": (MainMode.M1_FIXED_QUOTA, OrderingStrategy.O1_GLOBAL_PRIORITY),
        "C02": (MainMode.M1_FIXED_QUOTA, OrderingStrategy.O2_RESPONSE_RATIO),
        "C03": (MainMode.M1_FIXED_QUOTA, OrderingStrategy.O3_DEADLINE),
        "C04": (MainMode.M1_FIXED_QUOTA, OrderingStrategy.O4_FIFO),
        "C05": (MainMode.M2_REALTIME_PREEMPT, OrderingStrategy.O1_GLOBAL_PRIORITY),
        "C06": (MainMode.M2_REALTIME_PREEMPT, OrderingStrategy.O2_RESPONSE_RATIO),
        "C07": (MainMode.M2_REALTIME_PREEMPT, OrderingStrategy.O3_DEADLINE),
        "C08": (MainMode.M2_REALTIME_PREEMPT, OrderingStrategy.O4_FIFO),
        "C09": (MainMode.M3_DYNAMIC_WEIGHT, OrderingStrategy.O1_GLOBAL_PRIORITY),
        "C10": (MainMode.M3_DYNAMIC_WEIGHT, OrderingStrategy.O2_RESPONSE_RATIO),
        "C11": (MainMode.M3_DYNAMIC_WEIGHT, OrderingStrategy.O3_DEADLINE),
        "C12": (MainMode.M3_DYNAMIC_WEIGHT, OrderingStrategy.O4_FIFO),
    }
    
    if combination_id not in mapping:
        raise ValueError(f"Unknown combination ID: {combination_id}")
        
    mode, ordering = mapping[combination_id]
    return UnifiedScheduler(mode, ordering)
