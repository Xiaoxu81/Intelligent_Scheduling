import heapq
from src.models.task import Task
from src.models.resource import Resource
from src.environment.simulation import SimulationEnv, Event


def test_event_can_be_ordered_against_simpy_queue_entry():
    event = Event(2.0, "TASK_COMPLETION", "T1")
    assert event < (3.0, 0, object())

def test_basic_simulation():
    env = SimulationEnv()
    
    # 添加资源
    res = Resource("R1", "CPU")
    env.add_resource(res)
    
    # 添加任务
    task = Task("T1", priority=1, duration=10.0, arrival_time=5.0)
    env.add_task(task)
    
    print("Starting simulation...")
    
    # 步进到任务到达
    env.step()
    assert env.current_time == 5.0
    assert task.status.value == "READY"
    
    # 手动模拟调度决策：将 T1 分配给 R1
    # 在真实系统中，这将由调度器触发
    res.status = "BUSY"
    task.status = "RUNNING"
    task.start_time = env.current_time
    completion_time = env.current_time + task.duration
    heapq.heappush(env.event_queue, Event(completion_time, "TASK_COMPLETION", ("T1", "R1")))
    
    # 步进到任务完成
    env.step()
    assert env.current_time == 15.0
    assert task.status.value == "COMPLETED"
    
    print("Basic simulation test passed!")

if __name__ == "__main__":
    test_basic_simulation()
