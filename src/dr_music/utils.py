from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def sum_of_diagonals(matrix: torch.Tensor) -> torch.Tensor:
    """Return sums of all diagonals of a square matrix.

    The output order matches torch.diagonal offsets from -(N-1) to +(N-1).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square 2D tensor, got shape {tuple(matrix.shape)}")

    n = matrix.shape[0]
    # offsets: -(n-1), ..., 0, ..., +(n-1)
    offsets = torch.arange(-(n - 1), n, device=matrix.device, dtype=torch.int64)
    sums = [torch.sum(torch.diagonal(matrix, offset=int(k))) for k in offsets]
    return torch.stack(sums, dim=0)


def companion_eig_roots(coeff: torch.Tensor) -> torch.Tensor:
    """Compute roots of polynomial defined by coefficients using companion matrix eigenvalues.

    This mirrors the method in the original repo:
      A = diag(1, -1); A[0,:] = -coeff[1:]/coeff[0]
      roots = eigvals(A)

    Note: coeff[0] must be non-zero.
    """
    if coeff.ndim != 1:
        raise ValueError("coeff must be a 1D tensor")
    if coeff.numel() < 2:
        raise ValueError("coeff must have at least 2 elements")
    if torch.isclose(coeff[0], torch.tensor(0, dtype=coeff.dtype, device=coeff.device)):
        raise ValueError("coeff[0] must be non-zero")

    # degree = len(coeff)-1, companion matrix is (deg x deg)
    deg = coeff.numel() - 1
    a = torch.zeros((deg, deg), dtype=coeff.dtype, device=coeff.device)
    if deg > 1:
        a[1:, :-1] = torch.eye(deg - 1, dtype=coeff.dtype, device=coeff.device)
    a[0, :] = -coeff[1:] / coeff[0]
    return torch.linalg.eigvals(a)
