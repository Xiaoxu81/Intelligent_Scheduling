from typing import List, Tuple, Dict
from src.strategies.base import BaseScheduler
from src.models.task import Task, TaskStatus
from src.models.resource import Resource, ResourceStatus

class CompositeScheduler(BaseScheduler):
    """
    组合调度策略：支持多级队列与混合规则。
    对应 4.2.3 运行反馈驱动的多级队列策略以及跨策略组合。
    """
    def __init__(self, inter_queue_strategy: str = "PRIORITY"):
        self.inter_queue_strategy = inter_queue_strategy
        # 模拟多级队列：[实时队列, 高优先级队列, 普通队列, 低优先级队列]
        self.queues: List[List[Task]] = [[], [], [], []]
        self.wait_threshold = 20.0  # 防饥饿阈值
        self.execution_threshold = 10.0  # 防垄断阈值

    def _update_queues(self, ready_tasks: List[Task], current_time: float):
        """将就绪任务分配到对应队列，并处理 4.2.3 的优先级动态调整"""
        self.queues = [[], [], [], []]
        for task in ready_tasks:
            # 1. 实时任务进入最高级队列 (4.2.5)
            if getattr(task, 'is_realtime', False):
                self.queues[0].append(task)
                continue

            # 2. 计算等待时间（防饥饿 4.2.3a）
            wait_time = current_time - (task.wait_start_time or current_time)
            
            # 3. 计算执行行为（防垄断 4.2.3b）
            # 此处简化逻辑：根据初始优先级和动态反馈决定所属队列
            base_idx = self._get_base_queue_idx(task)
            
            if wait_time > self.wait_threshold and base_idx > 1:
                final_idx = base_idx - 1  # 提升优先级
                # print(f"[MLQ] Boosting task {task.task_id} due to starvation.")
            elif task.cpu_time_used > self.execution_threshold and base_idx < 3:
                final_idx = base_idx + 1  # 降低优先级
                # print(f"[MLQ] Lowering task {task.task_id} due to resource monopoly.")
            else:
                final_idx = base_idx
            
            self.queues[final_idx].append(task)

    def _get_base_queue_idx(self, task: Task) -> int:
        # 假设优先级 1-10 映射到不同队列
        if task.priority >= 8: return 1
        if task.priority >= 4: return 2
        return 3

    def schedule(self, ready_tasks: List[Task], resources: List[Resource], current_time: float) -> List[Tuple[Task, Resource]]:
        self._update_queues(ready_tasks, current_time)
        decisions = []
        idle_resources = [r for r in resources if r.status == ResourceStatus.IDLE]

        # 跨队列调度：按队列顺序分配资源
        for queue in self.queues:
            if not idle_resources:
                break
            
            # 队列内调度策略 (Intra-queue)
            # 实时队列采用 EDF，其他队列采用 FCFS/Priority
            if queue == self.queues[0]:
                sorted_tasks = sorted(queue, key=lambda t: t.deadline if t.deadline else float('inf'))
            else:
                sorted_tasks = sorted(queue, key=lambda t: t.arrival_time) # FCFS

            for task in sorted_tasks:
                if not idle_resources:
                    break
                res = idle_resources.pop(0)
                decisions.append((task, res))
        
        return decisions

    @property
    def is_preemptive(self) -> bool:
        return True
