from src.experiments.effect_dataset import save_effect_dataset
from src.experiments.effect_schema import EFFECT_KEYS, OutcomeRecord
from src.experiments.train_effect_predictor import train_effect_predictor


def test_effect_predictor_training_writes_checkpoint_and_metrics(tmp_path):
    records = []
    for index, strategy in enumerate(["C01", "C04", "C01", "C04"]):
        records.append(
            OutcomeRecord(
                strategy_id=strategy,
                feasible=True,
                metrics={key: float(index + offset + 1) for offset, key in enumerate(EFFECT_KEYS)},
                next_state={"pending_workload": 1.0},
                metadata={"initial_state_fingerprint": f"state-{index // 2}"},
                initial_state={
                    "system": [0.0] * 6,
                    "tasks": [[0.0] * 9] * 5,
                    "demands": [[0.0] * 6] * 5,
                    "resources": [[0.0] * 5] * 2,
                    "weights": [0.25] * 4,
                },
            )
        )
    dataset = save_effect_dataset(records, tmp_path / "effects.json")
    result = train_effect_predictor(dataset, tmp_path / "model", epochs=2, seed=3)
    assert (tmp_path / "model" / "effect_predictor.pt").exists()
    assert result["train_samples"] > 0
    assert result["mae"] >= 0.0
