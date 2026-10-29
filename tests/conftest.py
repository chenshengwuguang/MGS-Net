import pytest
import torch

from mgs_net.synthetic import create_synthetic_dataset


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def synthetic(tmp_path):
    return create_synthetic_dataset(tmp_path / "synthetic")
