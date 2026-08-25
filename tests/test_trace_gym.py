from src.environment.trace_gym import TraceSchedulingEnv
from src.experiments.alibaba_trace import AlibabaTrace
from src.models.resource import Resource
from src.models.task import Task, TaskStatus


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
    assert terminated is True
    assert isinstance(truncated, bool)


def test_trace_gym_rebases_absolute_trace_time_for_learning():
    trace = AlibabaTrace(
        tasks=[Task("j1:M1", priority=1, duration=2.0, arrival_time=150000.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )
    env = TraceSchedulingEnv(trace, max_queue_size=4, max_resource_size=2)

    env.reset(seed=3)

    assert env.sim.current_time == 0.0
    assert next(iter(env.sim.tasks.values())).arrival_time == 0.0


def test_trace_environment_maps_restricted_strategy_actions():
    trace = AlibabaTrace(
        tasks=[Task("t1", priority=1, duration=1.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "alibaba_cluster_trace_v2018"},
    )
    env = TraceSchedulingEnv(trace, strategy_ids=["C04", "C09"])

    assert env.action_space.n == 2
    env.reset(seed=0)
    env.step(0)
    assert env.sim.scheduler is not None


def test_trace_environment_rewards_shorter_completed_workload_more():
    def run(duration):
        trace = AlibabaTrace(
            tasks=[Task("t1", priority=1, duration=duration, arrival_time=0.0)],
            resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
            metadata={"data_source": "test"},
        )
        env = TraceSchedulingEnv(trace, strategy_ids=["C01"])
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step(0)
        assert terminated is True
        return reward

    assert run(1.0) - run(10.0) > 5.0


def test_trace_observation_exposes_pending_workload_signature():
    def state_for(duration):
        trace = AlibabaTrace(
            tasks=[Task("t1", priority=1, duration=duration, arrival_time=0.0)],
            resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
            metadata={"data_source": "test"},
        )
        env = TraceSchedulingEnv(trace, strategy_ids=["C01"])
        observation, _ = env.reset(seed=0)
        return observation["system"][5]

    assert state_for(10.0) > state_for(1.0)


def test_trace_environment_can_make_one_task_decision_at_a_time():
    trace = AlibabaTrace(
        tasks=[
            Task("t1", priority=1, duration=10.0, arrival_time=0.0),
            Task("t2", priority=1, duration=10.0, arrival_time=0.0),
        ],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "test"},
    )
    env = TraceSchedulingEnv(trace, strategy_ids=["C01"], max_assignments_per_step=1)
    env.reset(seed=0)
    env.step(0)

    statuses = [task.status for task in env.sim.tasks.values()]
    assert statuses.count(TaskStatus.RUNNING) == 1
    assert statuses.count(TaskStatus.READY) == 1


def test_trace_environment_can_inject_controlled_fault():
    trace = AlibabaTrace(
        tasks=[Task("t1", priority=1, duration=10.0, arrival_time=0.0)],
        resources=[Resource("m1", "Machine", capabilities={"machine": 2.0})],
        metadata={"data_source": "test"},
    )
    env = TraceSchedulingEnv(
        trace,
        strategy_ids=["C01"],
        max_assignments_per_step=1,
        fault_resource="m1",
        fault_at=0.5,
        fault_duration=2.0,
    )
    env.reset(seed=0)
    env.step(0)

    assert env.sim.resources["m1"].status.value == "FAULT"
