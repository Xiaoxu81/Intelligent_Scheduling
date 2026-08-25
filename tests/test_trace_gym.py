from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace
from src.models.resource import Resource
from src.models.task import Task


def test_trace_gym_uses_trace_tasks_and_report_state():
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=2.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capacity=1.0, capabilities={"machine": 2.0})],
        metadata={"data_source": "test"},
    )
    env = TraceSchedulingEnv(trace, max_queue_size=4, max_resource_size=2)

    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    assert len(env.sim.tasks) == 1
    assert len(env.sim.resources) == 1

    next_observation, reward, terminated, truncated, _ = env.step(0)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
