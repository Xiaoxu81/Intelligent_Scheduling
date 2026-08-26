"""Train and evaluate the context-conditioned strategy effect predictor."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from src.experiments.effect_dataset import load_effect_dataset
from src.experiments.effect_schema import EFFECT_KEYS
from src.models.effect_predictor import EffectPredictor


def _batch(records, strategy_to_index):
    return {
        "system": torch.tensor([r.initial_state["system"] for r in records], dtype=torch.float32),
        "tasks": torch.tensor([r.initial_state["tasks"] for r in records], dtype=torch.float32),
        "demands": torch.tensor([r.initial_state["demands"] for r in records], dtype=torch.float32),
        "resources": torch.tensor([r.initial_state["resources"] for r in records], dtype=torch.float32),
        "weights": torch.tensor([r.initial_state["weights"] for r in records], dtype=torch.float32),
        "strategy_id": torch.tensor([strategy_to_index[r.strategy_id] for r in records], dtype=torch.long),
    }


def _targets(records):
    return torch.tensor([[r.metrics[key] for key in EFFECT_KEYS] for r in records], dtype=torch.float32)


def _split(records, seed):
    groups = sorted({r.metadata.get("initial_state_fingerprint", str(i)) for i, r in enumerate(records)})
    random.Random(seed).shuffle(groups)
    if len(groups) < 2:
        return records, records
    cut = max(1, int(len(groups) * 0.8))
    train_groups = set(groups[:cut])
    train = [r for r in records if r.metadata.get("initial_state_fingerprint") in train_groups]
    test = [r for r in records if r.metadata.get("initial_state_fingerprint") not in train_groups]
    return train or records, test or records


def _predict(model, records, strategy_to_index, target_mean, target_std):
    with torch.no_grad():
        mean, _ = model(_batch(records, strategy_to_index))
    return mean * target_std + target_mean


def train_effect_predictor(dataset_path: str | Path, output_dir: str | Path, epochs: int = 50, seed: int = 0) -> Dict[str, float]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    records = load_effect_dataset(dataset_path)
    if not records:
        raise ValueError("effect dataset must not be empty")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_records, test_records = _split(records, seed)
    strategy_ids = sorted({record.strategy_id for record in records})
    strategy_to_index = {strategy: index for index, strategy in enumerate(strategy_ids)}
    train_target = _targets(train_records)
    target_mean = train_target.mean(dim=0)
    target_std = train_target.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_y = (train_target - target_mean) / target_std
    model = EffectPredictor(strategy_count=len(strategy_ids))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_batch = _batch(train_records, strategy_to_index)
    for _ in range(epochs):
        mean, log_variance = model(train_batch)
        loss = 0.5 * (torch.exp(-log_variance) * (train_y - mean).pow(2) + log_variance).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = _predict(model, test_records, strategy_to_index, target_mean, target_std)
    actual = _targets(test_records)
    error = predictions - actual
    result = {
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "mae": float(error.abs().mean()),
        "rmse": float(error.pow(2).mean().sqrt()),
        "final_loss": float(loss.detach()),
        "strategy_count": len(strategy_ids),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "strategy_ids": strategy_ids,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "metrics": result,
    }, output / "effect_predictor.pt")
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
