"""Context-conditioned predictor for candidate strategy effects."""

from __future__ import annotations

import torch
import torch.nn as nn


class EffectPredictor(nn.Module):
    """Predict mean and log variance for the six-effect outcome vector."""

    def __init__(
        self,
        strategy_count: int = 12,
        task_feat_dim: int = 9,
        demand_feat_dim: int = 6,
        system_feat_dim: int = 6,
        resource_feat_dim: int = 5,
        weight_feat_dim: int = 4,
        output_dim: int = 6,
        hidden_dim: int = 96,
    ):
        super().__init__()
        if strategy_count < 1:
            raise ValueError("strategy_count must be positive")
        self.strategy_embedding = nn.Embedding(strategy_count, 16)
        self.encoder = nn.Sequential(
            nn.Linear(task_feat_dim * 2 + demand_feat_dim * 2 + resource_feat_dim * 2 + system_feat_dim + weight_feat_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_variance_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        tasks = batch["tasks"]
        demands = batch["demands"]
        resources = batch["resources"]
        system = batch["system"]
        weights = batch["weights"]
        strategy_id = batch["strategy_id"].long()
        task_summary = torch.cat([tasks.mean(dim=1), tasks.max(dim=1).values], dim=1)
        demand_summary = torch.cat([demands.mean(dim=1), demands.max(dim=1).values], dim=1)
        resource_summary = torch.cat([resources.mean(dim=1), resources.max(dim=1).values], dim=1)
        strategy = self.strategy_embedding(strategy_id)
        hidden = self.encoder(torch.cat([task_summary, demand_summary, resource_summary, system, weights, strategy], dim=1))
        return self.mean_head(hidden), self.log_variance_head(hidden).clamp(-10.0, 10.0)
