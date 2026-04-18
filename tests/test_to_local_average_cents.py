from __future__ import annotations

import torch

from torch_rmve.core import to_local_average_cents, to_local_average_cents_old


def test_to_local_average_cents_equivalence_random() -> None:
    torch.manual_seed(0)
    salience = torch.rand(3, 17, 360, dtype=torch.float32)

    expected = to_local_average_cents_old(salience)
    actual = to_local_average_cents(salience)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_to_local_average_cents_equivalence_edge_bins() -> None:
    salience = torch.zeros(2, 4, 360, dtype=torch.float32)
    peak_bins = torch.tensor([[0, 1, 4, 8], [351, 355, 358, 359]])
    salience.scatter_(2, peak_bins.unsqueeze(-1), 1.0)
    salience += 0.01

    expected = to_local_average_cents_old(salience)
    actual = to_local_average_cents(salience)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
