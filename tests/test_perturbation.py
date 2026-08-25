from src.models.task import Task, TaskStatus
from src.models.resource import Resource, ResourceStatus
from src.environment.simulation import SimulationEnv
from src.environment.perturbation import PerturbationManager
from src.strategies.heuristics import get_scheduler_by_id

def test_complex_scenario():
    env = SimulationEnv(scheduler=get_scheduler_by_id("C01"))
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
    env.inject_resource_fault("R1", delay=2.0, duration=2.0)
    
    print("\n--- Starting Complex Scenario Simulation ---")
    
    # 执行仿真
    while env.event_queue:
        event = env.step()
        
    assert env.tasks["T1"].status == TaskStatus.FAILED
    assert env.resources["R1"].status == ResourceStatus.IDLE
    print("--- Complex Scenario Test Passed ---")

if __name__ == "__main__":
    test_complex_scenario()
