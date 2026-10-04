"""FinBERT query/value LoRA, following the Phase 17 frozen-base architecture.

Kept independent of legacy experiment imports and their sibling Bayesian dependency.
CoLoRA uses the supplied paper's shared B, A and task-specific Lambda: B Lambda_t A.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CollaborativeLoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=8, dropout=0.05, tasks=1):
        super().__init__()
        self.base_layer = base_layer
        for p in base_layer.parameters():
            p.requires_grad = False
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lambdas = nn.Parameter(torch.eye(rank).repeat(tasks, 1, 1)) if tasks > 1 else None
        self.task = 0

    def forward(self, x):
        h = F.linear(self.dropout(x), self.lora_a)
        if self.lambdas is not None:
            h = F.linear(h, self.lambdas[self.task])
        return self.base_layer(x) + self.scaling * F.linear(h, self.lora_b)


def inject_lora(model, config, tasks=1):
    for name, module in list(model.named_modules()):
        if "attention.self" in name:
            for key in ("query", "value"):
                layer = getattr(module, key, None)
                if isinstance(layer, nn.Linear):
                    setattr(module, key, CollaborativeLoRALinear(
                        layer, config["rank"], config["alpha"], config["dropout"], tasks))
    for name, param in model.named_parameters():
        if "lora_" in name or "lambdas" in name or "classifier" in name:
            param.requires_grad = True


def set_task(model, task):
    for layer in model.modules():
        if isinstance(layer, CollaborativeLoRALinear):
            layer.task = task
