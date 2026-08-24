from src.environment.gym_wrapper import SchedulingEnv


def test_observation_exposes_report_state_components():
    env = SchedulingEnv(max_queue_size=5, max_resource_size=3)
    observation, _ = env.reset(seed=13)

    assert observation["system"].shape == (6,)
    assert observation["tasks"].shape == (5, 9)
    assert observation["resources"].shape == (3, 5)
    assert observation["weights"].shape == (4,)
    assert observation["global"].shape == (2,)


def test_same_seed_produces_same_report_state():
    first, _ = SchedulingEnv(max_queue_size=5, max_resource_size=3).reset(seed=13)
    second, _ = SchedulingEnv(max_queue_size=5, max_resource_size=3).reset(seed=13)

    for key in ("system", "tasks", "resources", "weights"):
        assert (first[key] == second[key]).all()
