import torch

from dr_music.model import DeepRootNet


def test_forward_shape():
    model = DeepRootNet(tau=2)
    x = torch.randn(3, 2, 64, 64)  # large enough for conv/pool stack
    y = model(x)
    assert y.shape[0] == 3
