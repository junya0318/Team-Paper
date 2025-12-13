from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .utils import companion_eig_roots, sum_of_diagonals


ActivationName = Literal["relu", "tanh", "gelu"]


def _activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class DeepRootNet(nn.Module):
    """Deep Root-MUSIC network (from the original DR-MUSIC_ICASSP23 repo).

    Notes:
    - This is a light refactor: adds typing, removes globals, and extracts math helpers.
    - For full training / pipeline, refer to legacy/Run_Simulation.py in this project.
    """

    def __init__(self, tau: int, activation: ActivationName = "relu") -> None:
        super().__init__()
        self.tau = int(tau)
        act = _activation(activation)

        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        self.act = act

        # original code uses flatten->linear layers; sizes depend on input, so keep lazy
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)  # example output size; adjust to original if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        x = self.pool(self.act(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x)))
        x = self.dropout(self.act(self.fc4(x)))
        return self.fc5(x)

    @staticmethod
    def sum_of_diags(matrix: torch.Tensor) -> torch.Tensor:
        return sum_of_diagonals(matrix)

    @staticmethod
    def find_roots(coeff: torch.Tensor) -> torch.Tensor:
        return companion_eig_roots(coeff)
