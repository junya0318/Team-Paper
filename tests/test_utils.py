import numpy as np
import torch

from dr_music.utils import companion_eig_roots, sum_of_diagonals


def test_sum_of_diagonals_matches_numpy():
    a = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    got = sum_of_diagonals(a).cpu().numpy()

    # numpy: offsets -(n-1) .. (n-1)
    n = a.shape[0]
    expected = []
    for k in range(-(n-1), n):
        expected.append(np.trace(a.numpy(), offset=k))
    expected = np.array(expected, dtype=float)

    assert np.allclose(got, expected)


def test_companion_eig_roots_consistency():
    # polynomial: 2x^3 + 3x^2 - x + 5
    coeff = torch.tensor([2.0, 3.0, -1.0, 5.0])
    roots = companion_eig_roots(coeff).cpu().numpy()

    # Compare to numpy roots (order can differ)
    np_roots = np.roots(coeff.numpy())

    # sort by angle then magnitude for stable comparison
    def key(z):
        return (np.angle(z), np.abs(z))

    roots_sorted = np.array(sorted(roots, key=key))
    np_sorted = np.array(sorted(np_roots, key=key))
    assert np.allclose(roots_sorted, np_sorted, atol=1e-5)
