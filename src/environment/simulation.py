import simpy
from typing import List, Dict, Any, Optional
from src.models.task import Task, TaskStatus
from src.models.resource import Resource, ResourceStatus
from src.strategies.base import BaseScheduler


class Event:
    """领域事件记录（保持对外接口兼容）"""
    def __init__(self, time: float, event_type: str, data: Any):
        self.time = time
        self.event_type = event_type
        self.data = data

    def __lt__(self, other):
        other_time = other.time if isinstance(other, Event) else other[0]
        return self.time < other_time


class SimulationEnv:
    """
    基于 SimPy 的离散事件仿真引擎

    核心 SimPy 特性：
    - simpy.Environment       : 仿真时钟与事件调度
    - simpy.Process + timeout : 任务到达与执行建模
    - process.interrupt()     : 抢占式调度实现
    """

    def __init__(self, scheduler: Optional[BaseScheduler] = None):
        self.env = simpy.Environment()
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        self.scheduler = scheduler

        # 任务执行进程引用（用于 interrupt 实现抢占）
        self._task_processes: Dict[str, simpy.Process] = {}

        # 领域事件缓冲区（供 step() 逐个消费）
        self._domain_events: List[Event] = []

    # ========== 属性接口 ==========

    @property
    def current_time(self) -> float:
        return self.env.now

    @property
    def event_queue(self):
        """兼容接口：SimPy 内部事件队列，用于判断仿真是否结束"""
        return self.env._queue

    # ========== 实体注册 ==========

    def add_resource(self, resource: Resource):
        self.resources[resource.resource_id] = resource

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task
        # 启动 SimPy 任务到达进程
        self.env.process(self._task_arrival_process(task))

    # ========== 仿真推进 ==========

    def step(self):
        """推进仿真到下一个领域事件（任务到达/完成/故障/恢复）"""
        try:
            while self.env.peek() < float('inf'):
                self.env.step()
                if self._domain_events:
                    event = self._domain_events.pop(0)
                    self._run_scheduler()
                    return event
        except simpy.core.EmptySchedule:
            pass
        return None

    def run_until(self, end_time: float):
        """运行仿真直到指定时间"""
        while self.env.peek() < end_time:
            result = self.step()
            if result is None:
                break

    # ========== SimPy 进程定义 ==========

    def _task_arrival_process(self, task: Task):
        """
        SimPy 进程：模拟任务在指定时间到达
        使用 env.timeout() 延迟到达
        """
        delay = max(0.0, task.arrival_time - self.env.now)
        yield self.env.timeout(delay)

        # 检查依赖是否全部满足
        if all(self.tasks[dep_id].status == TaskStatus.COMPLETED
               for dep_id in task.dependencies):
            task.status = TaskStatus.READY
            print(f"[Time {self.env.now:.2f}] Task {task.task_id} arrived and is READY.")
        else:
            task.status = TaskStatus.PENDING
            print(f"[Time {self.env.now:.2f}] Task {task.task_id} arrived but is PENDING (waiting for dependencies).")

        self._domain_events.append(Event(self.env.now, "TASK_ARRIVAL", task.task_id))

    def _task_execution_process(self, task: Task, resource: Resource):
        """
        SimPy 进程：模拟任务执行
        利用 simpy.Interrupt 实现可中断的任务执行（抢占/故障）
        """
        try:
            yield self.env.timeout(task.remaining_time)

            # 正常完成
            task.cpu_time_used += task.remaining_time
            task.status = TaskStatus.COMPLETED
            task.end_time = self.env.now
            resource.status = ResourceStatus.IDLE
            resource.current_task_id = None
            print(f"[Time {self.env.now:.2f}] Task {task.task_id} completed on Resource {resource.resource_id}.")

            self._domain_events.append(
                Event(self.env.now, "TASK_COMPLETION", (task.task_id, resource.resource_id))
            )

            # 检查依赖于该任务的后续任务
            self._check_dependent_tasks(task.task_id)

        except simpy.Interrupt:
            # 被抢占或故障中断，状态由调用方（_preempt_task / _resource_fault_process）更新
            pass
        finally:
            if task.task_id in self._task_processes:
                del self._task_processes[task.task_id]

    def _resource_fault_process(self, resource_id: str, delay: float, duration: float):
        """
        SimPy 进程：资源故障与自动恢复
        delay    : 从当前时刻起延迟多久触发故障
        duration : 故障持续时长
        """
        yield self.env.timeout(delay)

        resource = self.resources[resource_id]
        resource.status = ResourceStatus.FAULT
        print(f"[Time {self.env.now:.2f}] RESOURCE FAULT: Resource {resource_id} is down.")

        # 中断正在执行的任务
        if resource.current_task_id:
            task_id = resource.current_task_id
            self.tasks[task_id].status = TaskStatus.FAILED
            resource.current_task_id = None
            if task_id in self._task_processes:
                self._task_processes[task_id].interrupt("fault")
            print(f"[Time {self.env.now:.2f}] Task {task_id} FAILED due to resource fault.")

        self._domain_events.append(Event(self.env.now, "RESOURCE_FAULT", resource_id))

        # 等待恢复
        yield self.env.timeout(duration)
        resource.status = ResourceStatus.IDLE
        print(f"[Time {self.env.now:.2f}] RESOURCE RECOVERY: Resource {resource_id} is back online.")

        self._domain_events.append(Event(self.env.now, "RESOURCE_RECOVERY", resource_id))

    # ========== 调度与抢占 ==========

    def _execute_task(self, task: Task, resource: Resource):
        """启动任务执行（创建 SimPy 执行进程）"""
        resource.status = ResourceStatus.BUSY
        resource.current_task_id = task.task_id
        task.status = TaskStatus.RUNNING
        task.start_time = self.env.now
        task.assigned_resource_id = resource.resource_id

        # 创建 SimPy 执行进程（可被 interrupt 中断）
        proc = self.env.process(self._task_execution_process(task, resource))
        self._task_processes[task.task_id] = proc
        print(f"[Time {self.env.now:.2f}] Scheduler: Assigned {task.task_id} to {resource.resource_id} (Remaining: {task.remaining_time:.2f})")

    def _preempt_task(self, task: Task, resource: Resource):
        """通过 SimPy interrupt 实现任务抢占"""
        print(f"[Time {self.env.now:.2f}] PREEMPTION: Stopping Task {task.task_id} on {resource.resource_id}")

        # 更新任务状态（使用 remaining_time 保存进度）
        elapsed = self.env.now - task.start_time
        task.remaining_time -= elapsed
        task.cpu_time_used += elapsed
        task.status = TaskStatus.READY
        task.start_time = None

        # 释放资源
        resource.status = ResourceStatus.IDLE
        resource.current_task_id = None

        # 通过 SimPy interrupt 中断执行进程
        if task.task_id in self._task_processes:
            self._task_processes[task.task_id].interrupt("preemption")

    def _run_scheduler(self):
        """运行调度器"""
        if not self.scheduler:
            return

        ready_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.READY]
        resources = list(self.resources.values())

        # 处理剥夺式抢占
        if self.scheduler.is_preemptive:
            self._handle_preemption()

        # 获取调度决策并执行
        decisions = self.scheduler.schedule(ready_tasks, resources, self.env.now)
        for task, resource in decisions:
            self._execute_task(task, resource)

    def _handle_preemption(self):
        """检查是否有更高优先级的任务可以抢占当前运行的任务"""
        ready_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.READY]
        if not ready_tasks:
            return

        for res_id, resource in self.resources.items():
            if resource.status == ResourceStatus.BUSY:
                running_task = self.tasks[resource.current_task_id]

                has_rt_ready = any(getattr(t, 'is_realtime', False) for t in ready_tasks)
                curr_is_rt = getattr(running_task, 'is_realtime', False)

                if has_rt_ready and not curr_is_rt:
                    self._preempt_task(running_task, resource)

    def _check_dependent_tasks(self, completed_task_id: str):
        """检查依赖于已完成任务的后续任务是否可以就绪"""
        for t_id, t in self.tasks.items():
            if completed_task_id in t.dependencies and t.status == TaskStatus.PENDING:
                if all(self.tasks[dep].status == TaskStatus.COMPLETED
                       for dep in t.dependencies):
                    t.status = TaskStatus.READY
                    print(f"[Time {self.env.now:.2f}] Task {t_id} is now READY (dependencies satisfied).")

    # ========== 外部注入接口 ==========

    def inject_resource_fault(self, resource_id: str, delay: float, duration: float):
        """注入资源故障事件（SimPy 进程）"""
        self.env.process(self._resource_fault_process(resource_id, delay, duration))

    def inject_task_arrival(self, task: Task, trigger_time: float):
        """在指定时间注入新任务"""
        task.arrival_time = trigger_time
        self.tasks[task.task_id] = task
        self.env.process(self._task_arrival_process(task))
