import numpy as np

from src.environment.gym_wrapper import SchedulingEnv


def test_observation_contains_continuous_task_demand_features():
    env = SchedulingEnv(max_queue_size=5, max_resource_size=2)
    observation, _ = env.reset(seed=7)
    assert observation["demands"].shape == (5, 6)
    assert np.all(observation["demands"] >= -1.0)
    assert np.all(observation["demands"] <= 1.0)
