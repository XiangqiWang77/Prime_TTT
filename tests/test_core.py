import torch
from torch import nn

from primettt.hypernetwork import PrimeTTTHyperNetwork
from primettt.lora import SlotInfo, stable_slot_seed
from eval.run_eval import contains_rate, first_mc_letter, token_f1


def test_stable_slot_seed_is_process_independent():
    assert stable_slot_seed("L31.q_proj") == stable_slot_seed("L31.q_proj")
    assert stable_slot_seed("L31.q_proj") != stable_slot_seed("L31.v_proj")


def test_hypernetwork_zero_init_returns_base():
    slots = [SlotInfo("L0.q_proj", 4, 6), SlotInfo("L0.v_proj", 4, 6)]
    hnet = PrimeTTTHyperNetwork(8, slots, rank=2, width=16, layers=1, heads=4)
    base = {"L0.q_proj": torch.ones(6, 2), "L0.v_proj": torch.ones(6, 2) * 2}
    hnet.set_B_base(base)
    out = hnet(torch.randn(1, 5, 8))
    assert torch.allclose(out["L0.q_proj"], base["L0.q_proj"])
    assert torch.allclose(out["L0.v_proj"], base["L0.v_proj"])


def test_scorers():
    assert contains_rate("The answer is Needle-42.", "needle 42") == 1.0
    assert first_mc_letter("I choose (C).") == "C"
    assert token_f1("red blue green", "red green") > 0.0

