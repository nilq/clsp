import pytest
import torch

from clsp.models import CLSP


@pytest.mark.unit_test
def test_clsp_runs() -> None:
    """Test that CLSP doesn't explode."""

    clsp = CLSP(text_mask_percentage=0.2, speech_mask_percentage=0.2)
    loss = clsp(
        torch.randint(0,256,(2,120)),
        torch.randint(0,8192,(2,250)),
        return_loss=True
    )

    assert 0.2 < loss < 1

    sim = clsp(
        torch.randint(0,256,(2,120)),
        torch.randint(0,8192,(2,250)),
        return_loss=True
    )

    assert 0.2 < sim < 1