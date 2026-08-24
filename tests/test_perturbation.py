import heapq
from src.models.task import Task, TaskStatus
from src.models.resource import Resource, ResourceStatus
from src.environment.simulation import SimulationEnv, Event
from src.environment.perturbation import PerturbationManager

def test_complex_scenario():
    env = SimulationEnv()
    pm = PerturbationManager(env)
    
    # 1. 设置资源
    res = Resource("R1", "Machine")
    env.add_resource(res)
    
    # 2. 设置带依赖的任务 T1 -> T2
    t1 = Task("T1", priority=1, duration=5.0, arrival_time=0.0)
    t2 = Task("T2", priority=1, duration=5.0, arrival_time=0.0, dependencies=["T1"])
    env.add_task(t1)
    env.add_task(t2)
    
    # 3. 调度一个确定的资源故障 (在时间 2.0 发生，持续 2.0)
    heapq.heappush(env.event_queue, Event(2.0, "RESOURCE_FAULT", "R1"))
    heapq.heappush(env.event_queue, Event(4.0, "RESOURCE_RECOVERY", "R1"))
    
    print("\n--- Starting Complex Scenario Simulation ---")
    
    # 执行仿真
    while env.event_queue:
        event = env.step()
        
        # 简单的手动调度逻辑逻辑
        ready_tasks = [t for t in env.tasks.values() if t.status == TaskStatus.READY]
        idle_resources = [r for r in env.resources.values() if r.status == ResourceStatus.IDLE]
        
        if ready_tasks and idle_resources:
            target_task = ready_tasks[0]
            target_res = idle_resources[0]
            
            target_res.status = ResourceStatus.BUSY
            target_res.current_task_id = target_task.task_id
            target_task.status = TaskStatus.RUNNING
            target_task.start_time = env.current_time
            
            completion_time = env.current_time + target_task.duration
            heapq.heappush(env.event_queue, Event(completion_time, "TASK_COMPLETION", (target_task.task_id, target_res.resource_id)))
            print(f"[Scheduler] Assigned {target_task.task_id} to {target_res.resource_id} at {env.current_time}")

    assert env.tasks["T2"].status == TaskStatus.COMPLETED
    print("--- Complex Scenario Test Passed ---")

if __name__ == "__main__":
    test_complex_scenario()
