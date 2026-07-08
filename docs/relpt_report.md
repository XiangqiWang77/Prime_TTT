# Prime_TTT RelPT Detailed Report

This report explains the RelPT component in Prime_TTT: what it is, how it fits
around PPO, which relational tables and SQL joins it uses, what the optimizer
controls, and how to read the latest PPO comparison.

## One-sentence summary

RelPT is not a new PPO loss. It is a relational control layer for
post-training preparation: prompts, trajectories, rewards, logprobs,
advantages, batches, policies, updates, and metrics are materialized as
append-only tables, so repeated PPO preparation can reuse existing rows and
only call expensive rollout, reward, or logprob executors for missing work.

## Figure 1: RelPT control layer

![RelPT pipeline](figures/relpt_pipeline.svg)

## Detailed illustration

In a normal PPO preparation loop, the system starts from a prompt batch, calls a
policy model to generate responses, scores each response with a reward function
or reward model, computes old-policy logprobs, builds advantages, and finally
hands a tensor batch to PPOTrainer. If the same policy step is retried, if a
run fails after generation but before training, if `K` rollouts per prompt are
increased, or if only rewards/logprobs need to be recomputed, a naive pipeline
often repeats work that has already been done.

RelPT inserts a relational layer before the trainer. The trainer remains a
black box; rollout, reward, and logprob systems also remain black-box
executors. RelPT only controls the artifacts that flow between them. Each
artifact is written into an append-only SQLite table with lineage keys such as
`prompt_id`, `policy_id`, `trajectory_id`, `reward_model_id`, and `batch_id`.
Before preparing a PPO batch, RelPT queries those tables to decide which rows
already exist and which rows are still missing.

The key idea is delta execution:

1. `prompt LEFT JOIN trajectory` identifies prompts that still need rollout rows for the current policy and requested number of responses.
2. `trajectory LEFT JOIN reward` identifies generated responses that still need scoring.
3. `reward_cache` reuses deterministic reward results across policy versions when the same prompt and response hash appear again.
4. `trajectory LEFT JOIN logprob` identifies responses that still need old policy logprob computation.
5. Completed trajectory, reward, logprob, and advantage rows are joined into a materialized PPO batch.
6. PPOTrainer consumes the same batch shape as the baseline path, so RelPT changes preparation and reuse, not the training loss.

## Tables and lineage

| table | role |
| --- | --- |
| `prompt` | Input prompt text, dataset metadata, and labels where available. |
| `policy` | Policy versions before and after PPO updates. |
| `trajectory` | Generated responses keyed by prompt and policy. |
| `reward` | Reward rows for specific trajectories and reward models. |
| `reward_cache` | Deterministic reward reuse keyed by reward model, prompt, and response hash. |
| `logprob` | Old-policy logprob rows needed by PPO. |
| `advantage` | Return and advantage rows used for training. |
| `batch` | Materialized PPO batch membership. |
| `update_record` | Lineage from input policy, batch, trainer backend, and output policy. |
| `metric` | Runtime, row-count, reward, and system metrics. |

## Latest TRL PPO comparison

- model: `sshleifer/tiny-gpt2`
- dataset: `pyarrow_arrow_cache`
- framework: `trl_ppo_0.11.4`
- steps: `200`
- batch size: `4`
- output: `runs/metrics/trl_ppo_relpt_17266929.json`

| metric | baseline | relpt | saved |
| --- | ---: | ---: | ---: |
| total_ms | 20576.0 | 12201.1 | 40.7% |
| prep_ms | 14955.4 | 7580.0 | 49.3% |
| generate_ms | 14887.2 | 7516.4 | 49.5% |
| reward_ms | 68.3 | 40.5 | 40.6% |
| generated_rows | 1600 | 800 | 50.0% |
| reward_rows | 1600 | 800 | 50.0% |

![TRL PPO result](figures/trl_ppo_result.svg)

## Complete SQLite RelPT prototype result

The full SQLite prototype saved 50.0% of
rollout rows, 99.3% of reward executor rows,
and 50.0% of logprob rows at 200 steps.
Training rows were unchanged because RelPT intentionally leaves PPO itself as
the same black-box update.

![Simulated RelPT result](figures/simulated_relpt_result.svg)

## How to read the result

The result supports a systems claim: RelPT can reduce duplicated PPO batch
preparation work while keeping the PPO trainer path unchanged. It does not
show that `sshleifer/tiny-gpt2` learned GSM8K math, because
`final_eval_reward_delta` and `mean_train_reward_delta` are both `0.0` in this
run. The useful signal is executor efficiency: with the same PPOTrainer, the
same model, and the same dataset, RelPT reduces generated rows, reward rows,
and preparation time.

PDF and HTML copies are generated from `docs/generate_relpt_report.py`.
