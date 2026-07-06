# PrimeTTT

PrimeTTT is a clean implementation scaffold for hypernetwork-warm-started
test-time training:

1. **Stage 1** fits per-example oracle LoRA `B*` with answer-conditioned NTP.
2. **Stage 2** trains `H_psi` to predict `Delta B = B* - B_base`.
3. **Stage 3** optionally fine-tunes `H_psi` through a differentiable PM-NTP
   inner loop, with an exact-MAML/FOMAML switch.
4. **Inference** runs `H_psi`, performs `K=8` PM-NTP polish steps, generates
   deterministically, then discards the LoRA state.

The old experiment scripts are kept at repository root for traceability.  New
code should import from `primettt/`.

## Local Assets

This repo is wired to the existing NTK_TTT layout on the Roberts cluster:

- Models: `../Models/Llama-3.1-8B-Instruct`, `../Models/Llama-3.2-3B-Instruct`,
  `../Models/Qwen3-4B-0725-Ins`, `../Models/gpt-oss-20b`.
- Datasets: `../archive/combined/niah_free/long_niah_free_{train,test}.jsonl`
  and OOD v2 files for HotpotQA, 2WikiMHQA, and MuSR.
- Path aliases are recorded in `data/paths.py`; the runnable local config is
  `configs/local_llama8b_niah_stage1.yaml`.

## Install

```bash
pip install -e ".[dev]"
```

## Minimal Example

Build a small synthetic NIAH file:

```bash
python -m data.build_niah --out data/splits/niah_smoke.jsonl --n 8 --context-tokens 512
```

Fit Stage 1 oracle adapters:

```bash
python -m primettt.stage1_oracles \
  --data data/splits/niah_smoke.jsonl \
  --model /nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Models/Llama-3.1-8B-Instruct \
  --out oracle_dumps/llama8b_niah_smoke \
  --max-position 32768 \
  --max-prompt-tokens 4096 \
  --steps 200
```

Warm up the hypernetwork:

```bash
python -m primettt.stage2_warmup \
  --oracle-dir oracle_dumps/llama8b_niah_smoke \
  --model /nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Models/Llama-3.1-8B-Instruct \
  --out release/llama8b_niah_smoke_hpsi.pt \
  --epochs 1
```

Run Stage 3 exact MAML, or add `--first-order` for the FOMAML ablation:

```bash
python -m primettt.stage3_maml \
  --oracle-dir oracle_dumps/llama8b_niah_smoke \
  --model /nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Models/Llama-3.1-8B-Instruct \
  --warmup-ckpt release/llama8b_niah_smoke_hpsi.pt \
  --out release/llama8b_niah_smoke_hpsi_stage3.pt \
  --inner-steps 8
```

Run deterministic test-time inference:

```bash
python -m primettt.inference \
  --data data/splits/niah_smoke.jsonl \
  --model /nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT/Models/Llama-3.1-8B-Instruct \
  --ckpt release/llama8b_niah_smoke_hpsi.pt \
  --out predictions/niah_smoke.jsonl \
  --inner-steps 8 \
  --inner-lr 0.001
```

Score contains-rate:

```bash
python -m eval.run_eval --predictions predictions/niah_smoke.jsonl --mode contains
```

## Reproducibility Notes

- All table rows, including frozen baselines, should load the same YaRN-extended
  base model through `primettt.model_loader`.
- LoRA `A` is fixed and generated from stable per-slot seeds.  Release bundles
  must include `H_psi`, `B_base`, and the slot manifest/seed configuration.
- Hypernetwork output heads are zero-initialized, so an untrained model returns
  exactly `B_base`.
- Evaluation uses deterministic generation and paired scorers so McNemar
  fix/break statistics are meaningful.
