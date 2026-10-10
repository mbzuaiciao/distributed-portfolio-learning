"""Neural network architectures for the collaborative synthetic experiment.

Implements SmallMLP and CoLoRAMLP as specified in
docs/bayesian_collaboration_design_amendment.md.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SmallMLP(nn.Module):
    """Standard 3-layer MLP for single-task and pooled baselines.

    Architecture:
        Input: 40 dims (20 features + 20 observation mask)
        Layer 1: Linear(40 -> 32), ReLU
        Layer 2: Linear(32 -> 16), ReLU
        Layer 3: Linear(16 -> 1), Sigmoid
    """

    def __init__(self, in_features: int = 40):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h).squeeze(-1)
        return torch.sigmoid(logits)


class CoLoRALinear(nn.Module):
    """Linear layer augmented with collaborative LoRA (shared B, A and per-task Lambda).

    Delta W = (alpha / rank) * B @ Lambda_task @ A
    """

    def __init__(
        self,
        in_features: int = 40,
        out_features: int = 32,
        rank: int = 4,
        alpha: float = 8.0,
        tasks: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.tasks = tasks

        # Base trainable linear projection
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.base_bias, -bound, bound)

        # LoRA parameters
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        # Per-task Lambda matrices
        self.lambdas = nn.Parameter(torch.eye(rank).unsqueeze(0).repeat(tasks, 1, 1))

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        # Base forward
        h_base = F.linear(x, self.base_weight, self.base_bias)
        # LoRA forward: x @ A.T @ Lambda.T @ B.T
        h_a = F.linear(x, self.lora_a)
        h_lambda = F.linear(h_a, self.lambdas[task_id])
        h_lora = F.linear(h_lambda, self.lora_b)
        return h_base + self.scaling * h_lora


class CoLoRAMLP(nn.Module):
    """Multi-task MLP with task-specific low-rank routing on the first layer."""

    def __init__(
        self,
        in_features: int = 40,
        rank: int = 4,
        alpha: float = 8.0,
        tasks: int = 3,
    ):
        super().__init__()
        self.colora_fc1 = CoLoRALinear(
            in_features=in_features,
            out_features=32,
            rank=rank,
            alpha=alpha,
            tasks=tasks,
        )
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        h = F.relu(self.colora_fc1(x, task_id=task_id))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h).squeeze(-1)
        return torch.sigmoid(logits)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}
