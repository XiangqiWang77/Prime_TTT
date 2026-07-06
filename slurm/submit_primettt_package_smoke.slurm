#!/bin/bash
# Lightweight validation for the cleaned primettt package on one RTX 6000 GPU.
#
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=primettt_pkg_smoke
#SBATCH --output=/nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Prime_TTT/logs/primettt_pkg_smoke_%j.out
#SBATCH --error=/nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Prime_TTT/logs/primettt_pkg_smoke_%j.err

set -eo pipefail

ROOT=/nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Prime_TTT
mkdir -p "$ROOT/logs"
source /home/da839/anaconda3/etc/profile.d/conda.sh
conda activate verl_custom
cd "$ROOT"

PYTHONPYCACHEPREFIX=/tmp/primettt_pycache python - <<'PY'
import torch

from eval.run_eval import contains_rate, first_mc_letter
from primettt.hypernetwork import PrimeTTTHyperNetwork
from primettt.inner_loop import differentiable_sgd_unroll
from primettt.lora import SlotInfo, stable_slot_seed

assert torch.cuda.is_available(), "CUDA is not visible"
device = torch.device("cuda")

assert stable_slot_seed("L31.q_proj") == stable_slot_seed("L31.q_proj")
assert stable_slot_seed("L31.q_proj") != stable_slot_seed("L31.v_proj")

slots = [SlotInfo("L0.q_proj", 4, 6), SlotInfo("L0.v_proj", 4, 6)]
hnet = PrimeTTTHyperNetwork(8, slots, rank=2, width=16, layers=1, heads=4).to(device)
base = {
    "L0.q_proj": torch.ones(6, 2, device=device),
    "L0.v_proj": torch.ones(6, 2, device=device) * 2,
}
hnet.set_B_base(base)
out = hnet(torch.randn(1, 5, 8, device=device))
assert torch.allclose(out["L0.q_proj"], base["L0.q_proj"])
assert torch.allclose(out["L0.v_proj"], base["L0.v_proj"])

params = {"x": torch.tensor([3.0], device=device, requires_grad=True)}
fast = differentiable_sgd_unroll(
    params,
    lambda p: (p["x"] ** 2).sum(),
    steps=2,
    lr=0.1,
    first_order=False,
)
assert fast["x"].item() < params["x"].item()

assert contains_rate("The answer is Needle-42.", "needle 42") == 1.0
assert first_mc_letter("I choose (C).") == "C"
print("PRIMETTT_PACKAGE_SMOKE_OK")
PY
