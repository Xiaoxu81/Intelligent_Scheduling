import torch

from src.models.effect_predictor import EffectPredictor


def test_effect_predictor_returns_multimetric_mean_and_uncertainty():
    model = EffectPredictor(strategy_count=3)
    batch = {
        "system": torch.zeros(2, 6),
        "tasks": torch.zeros(2, 5, 9),
        "demands": torch.zeros(2, 5, 6),
        "resources": torch.zeros(2, 2, 5),
        "weights": torch.full((2, 4), 0.25),
        "strategy_id": torch.tensor([0, 2]),
    }
    mean, log_variance = model(batch)
    assert mean.shape == (2, 6)
    assert log_variance.shape == (2, 6)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_variance).all()
