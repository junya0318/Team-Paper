from __future__ import annotations

import argparse

import torch

from .model import DeepRootNet
from .seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny demo for DeepRootNet packaging + reproducibility.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    model = DeepRootNet(tau=args.tau).to(device)
    x = torch.randn(4, args.tau, 16, 16, device=device)  # (batch, tau, H, W)
    y = model(x)
    print("output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
