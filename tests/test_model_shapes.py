import torch
from train_heads import HieraMIL3
from improve_batched import HieraMILB


def test_case_model_shapes():
    model = HieraMIL3(768)
    x = torch.randn(4, 768)
    vi = torch.tensor([0, 1, 0, 1])
    si = torch.tensor([0, 0, 1, 1])
    lb, ld, lq = model(x, vi, si)
    assert lb.shape == (3,)
    assert ld.shape == (4,)
    assert set(lq) == {0, 1}
    assert lq[0].shape == (5,)


def test_case_model_handles_three_views():
    model = HieraMIL3(768)
    x = torch.randn(3, 768)
    vi = torch.tensor([0, 1, 0])
    si = torch.tensor([0, 0, 1])
    lb, ld, lq = model(x, vi, si)
    assert lb.shape == (3,)
    assert lq[1].shape == (5,)


def test_batched_model_masks():
    model = HieraMILB(768)
    x = torch.randn(2, 4, 768)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    vi = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 0]])
    si = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
    lb, ld, lq, qm = model(x, mask, vi, si)
    assert lb.shape == (2, 3)
    assert ld.shape == (2, 4)
    assert lq.shape == (2, 2, 5)
    assert qm.tolist() == [[True, True], [True, False]]
