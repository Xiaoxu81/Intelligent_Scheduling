import random
from src.environment.simulation import SimulationEnv
from src.models.task import Task


class PerturbationManager:
    """
    动态扰动管理器
    利用 SimPy 进程模型生成随机干扰事件（资源故障、紧急任务注入）
    """
    def __init__(self, env: SimulationEnv):
        self.env = env

    def schedule_random_faults(
        self,
        resource_id: str,
        fault_rate: float,
        duration_range: tuple,
        end_time: float
    ):
        """
        为指定资源调度随机故障事件

        基于泊松过程（指数分布间隔）生成故障序列，
        每个故障通过 SimPy 进程建模，自动处理故障与恢复。

        Parameters:
            resource_id   : 目标资源 ID
            fault_rate    : 故障发生的概率密度（泊松过程 λ）
            duration_range: (min_duration, max_duration) 故障持续时长范围
            end_time      : 故障生成截止时间
        """
        current_t = 0.0
        while current_t < end_time:
            # 使用指数分布模拟故障间隔时间 (Poisson Process)
            interval = random.expovariate(fault_rate)
            current_t += interval
            if current_t >= end_time:
                break

            # 计算故障持续时长
            duration = random.uniform(*duration_range)

            # 通过 SimPy 进程注入故障（delay 为从仿真起始到故障发生的绝对时间）
            self.env.inject_resource_fault(resource_id, current_t, duration)

            # 更新时间指针到恢复之后，确保故障不重叠
            current_t += duration

    def inject_urgent_task(self, task_generator_func, trigger_time: float):
        """
        在特定时间注入紧急任务

        Parameters:
            task_generator_func: 返回 Task 实例的工厂函数
            trigger_time       : 任务注入时间
        """
        task = task_generator_func()
        self.env.inject_task_arrival(task, trigger_time)
