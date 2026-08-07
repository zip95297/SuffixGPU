from __future__ import annotations

import pytest
import torch


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.fixture(params=_devices())
def device(request) -> torch.device:
    return torch.device(request.param)
