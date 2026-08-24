from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from src.models.task import Task
from src.models.resource import Resource

class MainMode(Enum):
    M1_FIXED_QUOTA = "FIXED_QUOTA"
    M2_REALTIME_PREEMPT = "REALTIME_PREEMPT"
    M3_DYNAMIC_WEIGHT = "DYNAMIC_WEIGHT"

class OrderingStrategy(Enum):
    O1_GLOBAL_PRIORITY = "GLOBAL_PRIORITY"
    O2_RESPONSE_RATIO = "RESPONSE_RATIO"
    O3_DEADLINE = "DEADLINE"
    O4_FIFO = "FIFO"

class BaseScheduler(ABC):
    """调度策略基类"""
    def __init__(self, mode: MainMode, ordering: OrderingStrategy):
        self.mode = mode
        self.ordering = ordering

    @abstractmethod
    def schedule(
        self, 
        ready_tasks: List[Task], 
        resources: List[Resource], 
        current_time: float
    ) -> List[Tuple[Task, Resource]]:
        pass

    @property
    def is_preemptive(self) -> bool:
        # M2 和 M3 模式通常支持抢占
        return self.mode in [MainMode.M2_REALTIME_PREEMPT, MainMode.M3_DYNAMIC_WEIGHT]
