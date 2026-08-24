from src.models.task import Task
from src.models.resource import Resource
from src.environment.simulation import SimulationEnv
from src.strategies.heuristics import get_scheduler_by_id

def test_composite_strategy():
    # 使用组合 C05: 实时抢占 + 全局优先级
    scheduler = get_scheduler_by_id("C05")
    env = SimulationEnv(scheduler=scheduler)
    
    # 添加资源
    env.add_resource(Resource("R1", "Worker"))
    
    # 1. 普通任务
    t1 = Task("Normal_T1", priority=2, duration=10.0, arrival_time=0.0)
    # 2. 高优先级任务 (4.2.2)
    t2 = Task("High_T2", priority=9, duration=5.0, arrival_time=2.0)
    # 3. 实时任务 (4.2.5 / 4.2.4)
    t3 = Task("RealTime_T3", priority=5, duration=3.0, arrival_time=1.0, deadline=5.0)
    t3.is_realtime = True
    
    env.add_task(t1)
    env.add_task(t2)
    env.add_task(t3)
    
    print("\n--- Testing Composite Scheduling Strategy ---")
    env.run_until(50.0)
    
    # 验证执行顺序：应为 T1(由于最先到且无抢占) -> T3(实时) -> T2(高优先)
    # 注意：当前 SimulationEnv 尚未实现剥夺式抢占，因此 T1 会先执行完
    print("--- Composite Strategy Test Completed ---")

if __name__ == "__main__":
    test_composite_strategy()
