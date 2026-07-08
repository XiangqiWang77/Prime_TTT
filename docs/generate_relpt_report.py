#!/usr/bin/env python3
"""Generate the English Prime_TTT / RelPT report assets.

The script intentionally depends only on the Python standard library plus the
system `rsvg-convert` binary for PDF assembly.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
PAGES = DOCS / "report_pages"
W, H = 1240, 1754
M = 72
FONT = "DejaVu Sans, Arial, sans-serif"


def load_json(path: str) -> Dict[str, object]:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


TRL = load_json("runs/metrics/trl_ppo_relpt_17266929.json")
LOCAL = load_json("runs/metrics/relpt_gsm8k_200step_local.json")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: object) -> str:
    return f"{float(value):.1f}%"


def seconds(ms: object) -> str:
    return f"{float(ms) / 1000.0:.1f}s"


def wrap(text: str, chars: int) -> List[str]:
    out: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            out.append("")
        else:
            out.extend(textwrap.wrap(para, width=chars, break_long_words=False))
    return out


class Svg:
    def __init__(self, eyebrow: str):
        self.parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            '<rect width="100%" height="100%" fill="#fbfaf7"/>',
            (
                "<style>"
                f"text{{font-family:{FONT};fill:#1f2933}}"
                ".muted{fill:#5f6978}.mono{font-family:DejaVu Sans Mono,monospace}"
                "</style>"
            ),
        ]
        self.text(M, 62, eyebrow, 23, "#697386")
        self.line(M, 82, W - M, 82, "#d7dbe2", 2)

    def text(
        self,
        x: float,
        y: float,
        value: object,
        size: int = 28,
        fill: str = "#1f2933",
        weight: str = "400",
        anchor: str = "start",
        cls: str = "",
    ) -> None:
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" class="{cls}">{esc(value)}</text>'
        )

    def multiline(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 28,
        chars: int = 50,
        gap: int = 38,
        fill: str = "#1f2933",
        weight: str = "400",
    ) -> float:
        yy = y
        for line in wrap(text, chars):
            if not line:
                yy += gap // 2
            else:
                self.text(x, yy, line, size, fill, weight)
                yy += gap
        return yy

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "#ffffff",
        stroke: str = "#d9dee7",
        rx: int = 8,
        sw: int = 2,
    ) -> None:
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#9098a5", sw: int = 2) -> None:
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#596579") -> None:
        self.line(x1, y1, x2, y2, stroke, 3)
        if abs(x2 - x1) >= abs(y2 - y1):
            sign = 1 if x2 > x1 else -1
            self.parts.append(f'<path d="M{x2:.1f},{y2:.1f} l{-16 * sign},-9 l0,18 z" fill="{stroke}"/>')
        else:
            sign = 1 if y2 > y1 else -1
            self.parts.append(f'<path d="M{x2:.1f},{y2:.1f} l-9,{-16 * sign} l18,0 z" fill="{stroke}"/>')

    def footer(self, page: int) -> None:
        self.line(M, H - 74, W - M, H - 74, "#d7dbe2", 2)
        self.text(M, H - 35, "Prime_TTT RelPT report generated from this repository", 21, "#697386")
        self.text(W - M, H - 35, str(page), 21, "#697386", anchor="end")

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def bar(svg: Svg, x: float, y: float, label: str, base: float, relpt: float, unit: str, maxv: float) -> None:
    svg.text(x, y + 30, label, 25, weight="700")
    bx, bw = x + 255, 560
    svg.rect(bx, y, bw, 34, "#edf1f5", "#edf1f5", 4, 0)
    svg.rect(bx, y, bw * base / maxv, 34, "#d86c59", "#d86c59", 4, 0)
    svg.text(bx + bw + 22, y + 27, f"baseline {base:.1f}{unit}", 22, "#5e6978")
    svg.rect(bx, y + 50, bw, 34, "#edf1f5", "#edf1f5", 4, 0)
    svg.rect(bx, y + 50, bw * relpt / maxv, 34, "#2f8f83", "#2f8f83", 4, 0)
    svg.text(bx + bw + 22, y + 77, f"RelPT {relpt:.1f}{unit}", 22, "#5e6978")


def node(svg: Svg, x: float, y: float, title: str, body: str, fill: str = "#ffffff", w: int = 238, h: int = 118) -> None:
    svg.rect(x, y, w, h, fill, "#cbd3df")
    svg.text(x + w / 2, y + 44, title, 24, weight="700", anchor="middle")
    svg.text(x + w / 2, y + 78, body, 20, "#697386", anchor="middle")


def page_01() -> Path:
    s = Svg("Prime_TTT / RelPT")
    s.text(M, 160, "RelPT: reusable PPO preparation", 48, weight="700")
    s.text(M, 218, "through relational artifact control", 42, weight="700")
    s.multiline(
        M,
        295,
        "RelPT is the system layer in Prime_TTT that makes post-training artifacts reusable. "
        "It records prompts, rollouts, rewards, logprobs, advantages, batches, policies, and "
        "updates as append-only relational rows, then computes only the missing delta before PPO.",
        31,
        52,
        44,
    )
    s.rect(M, 555, W - 2 * M, 320, "#ffffff", "#d4dae3")
    s.text(M + 34, 615, "The short version", 34, weight="700")
    s.multiline(
        M + 34,
        675,
        "Baseline preparation redoes generation, reward scoring, and logprob work. RelPT first "
        "queries the artifact tables. Existing rows are reused; only absent rows trigger expensive "
        "executors. PPOTrainer still receives the same batch shape.",
        30,
        55,
        43,
    )
    s.text(M, 975, "What this report illustrates", 37, weight="700")
    items = [
        "Where RelPT sits around rollout, reward, logprob, and PPOTrainer.",
        "How SQL joins turn retries and partial recomputation into delta work.",
        "Which tables carry lineage for reproducibility and recovery.",
        "What the latest same-framework TRL PPO measurement shows.",
        "Why the result is a systems efficiency signal, not a model-quality claim.",
    ]
    y = 1040
    for item in items:
        s.text(M + 22, y, "- " + item, 28)
        y += 55
    s.footer(1)
    p = PAGES / "page_01.svg"
    s.save(p)
    return p


def page_02() -> Path:
    s = Svg("Baseline PPO preparation")
    s.text(M, 160, "Without RelPT: retries repeat work", 45, weight="700")
    boxes = [
        (M, 280, "Prompts", "GSM8K batch"),
        (M + 285, 280, "Policy", "tiny-gpt2"),
        (M + 570, 280, "Rollout", "generate"),
        (M + 855, 280, "Reward", "score"),
        (M + 570, 510, "Logprob", "old policy"),
        (M + 855, 510, "PPO batch", "tensors"),
        (M + 855, 740, "PPOTrainer", "same update"),
    ]
    for args in boxes:
        node(s, *args)
    s.arrow(M + 238, 339, M + 285, 339)
    s.arrow(M + 523, 339, M + 570, 339)
    s.arrow(M + 808, 339, M + 855, 339)
    s.arrow(M + 689, 398, M + 689, 510)
    s.arrow(M + 974, 398, M + 974, 510)
    s.arrow(M + 974, 628, M + 974, 740)
    s.rect(M, 1025, W - 2 * M, 330, "#fff7ed", "#efc58d")
    s.text(M + 34, 1088, "Why this wastes work", 34, weight="700")
    s.multiline(
        M + 34,
        1150,
        "A failed run, retry, cache miss investigation, or change from K=4 to K=6 can ask for "
        "almost the same batch again. A naive pipeline has no durable artifact map, so it often "
        "regenerates responses and rescans rewards even when most rows already exist.",
        30,
        55,
        43,
    )
    s.footer(2)
    p = PAGES / "page_02.svg"
    s.save(p)
    return p


def page_03() -> Path:
    s = Svg("RelPT dataflow")
    s.text(M, 155, "With RelPT: query first, execute only the delta", 45, weight="700")
    xs = [M, M + 285, M + 570, M + 855]
    titles = [("prompt", "input rows"), ("trajectory", "responses"), ("reward", "scores"), ("logprob", "PPO refs")]
    for i, (title, sub) in enumerate(titles):
        node(s, xs[i], 270, title, sub, "#f7fbff")
        if i:
            s.arrow(xs[i - 1] + 238, 329, xs[i], 329, "#5b7799")
    node(s, M + 285, 500, "reward_cache", "semantic reuse", "#f6fff8")
    node(s, M + 570, 500, "advantage", "returns", "#f8f5ff")
    node(s, M + 855, 500, "batch", "materialized", "#fffaf0")
    s.arrow(M + 404, 500, M + 690, 388, "#5f9b6a")
    s.arrow(M + 690, 500, M + 690, 388, "#8068b5")
    s.arrow(M + 974, 500, M + 974, 388, "#9f7f39")
    s.text(M, 760, "Delta planner worklists", 38, weight="700")
    rows = [
        ("rollout_requests", "prompts with too few usable trajectories for the current policy"),
        ("reward_traj_ids", "trajectories whose reward is not found in reward or reward_cache"),
        ("logprob_traj_ids", "trajectories missing old-policy logprob rows"),
        ("advantage_traj_ids", "rewarded trajectories missing returns and advantages"),
    ]
    y = 830
    for name, desc in rows:
        s.rect(M, y - 38, 280, 64, "#e8eef7", "#c9d4e5")
        s.text(M + 140, y + 4, name, 23, weight="700", anchor="middle", cls="mono")
        s.text(M + 320, y + 2, desc, 27, "#4b5563")
        y += 112
    s.footer(3)
    p = PAGES / "page_03.svg"
    s.save(p)
    return p


def page_04() -> Path:
    s = Svg("Tables and joins")
    s.text(M, 155, "Tables make lineage, reuse, and recovery explicit", 43, weight="700")
    rows = [
        ("prompt", "input text, dataset metadata, compact labels"),
        ("policy", "policy versions before and after PPO updates"),
        ("trajectory", "rollout artifacts keyed by prompt and policy"),
        ("reward", "score for a trajectory under a reward model"),
        ("reward_cache", "deterministic reuse by reward model, prompt, response hash"),
        ("logprob", "old-policy logprob rows required by PPO"),
        ("advantage", "return and advantage rows"),
        ("batch", "materialized PPO batch membership"),
        ("update_record", "input policy, output policy, batch, trainer lineage"),
        ("metric", "runtime, row-count, reward, and system metrics"),
    ]
    y = 230
    for i, (name, desc) in enumerate(rows):
        fill = "#ffffff" if i % 2 == 0 else "#f3f6fa"
        s.rect(M, y, 270, 58, fill, "#d6dce5", 2, 1)
        s.rect(M + 270, y, W - 2 * M - 270, 58, fill, "#d6dce5", 2, 1)
        s.text(M + 22, y + 38, name, 24, weight="700", cls="mono")
        s.text(M + 300, y + 38, desc, 24)
        y += 58
    s.rect(M, 930, W - 2 * M, 430, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 990, "Join semantics", 34, weight="700")
    s.multiline(
        M + 34,
        1050,
        "The important join is not only by trajectory id. reward_cache joins on reward_model_id, "
        "prompt_id, and response_hash, so deterministic rewards can be reused across policy "
        "versions when the same prompt-response pair appears again. Other LEFT JOINs identify "
        "missing rollouts, rewards, logprobs, and advantages.",
        29,
        58,
        42,
    )
    s.footer(4)
    p = PAGES / "page_04.svg"
    s.save(p)
    return p


def page_05() -> Path:
    s = Svg("Implementation map")
    s.text(M, 155, "Two implementations, one control-layer idea", 45, weight="700")
    s.rect(M, 250, W - 2 * M, 420, "#ffffff", "#d4dae3")
    s.text(M + 34, 310, "Full SQLite RelPT prototype", 34, weight="700")
    s.multiline(
        M + 34,
        370,
        "`relpt.py` creates the append-only schema and implements `RelPT.plan_ppo_batch` plus "
        "`RelPT.prepare_ppo_batch`. Rollout, reward, logprob, and PPO training are black-box "
        "executors, which makes the relational contract testable without GPU LLM frameworks.",
        29,
        58,
        42,
    )
    s.rect(M, 760, W - 2 * M, 430, "#ffffff", "#d4dae3")
    s.text(M + 34, 820, "Same-framework TRL PPO comparison", 34, weight="700")
    s.multiline(
        M + 34,
        880,
        "`trl_gsm8k_relpt_ppo.py` keeps both arms on the same TRL PPOTrainer path. The RelPT "
        "arm caches prepared query, response, and reward rows for retry reuse; the baseline "
        "prepares them again. This isolates preparation efficiency from trainer changes.",
        29,
        58,
        42,
    )
    s.rect(M, 1280, W - 2 * M, 160, "#f6fff8", "#92c9a2")
    s.text(M + 34, 1340, "Boundary", 32, weight="700")
    s.text(M + 34, 1392, "RelPT changes artifact planning and reuse. It does not change PPO loss.", 29)
    s.footer(5)
    p = PAGES / "page_05.svg"
    s.save(p)
    return p


def page_06() -> Path:
    s = Svg("TRL PPO measurement")
    s.text(M, 150, "200-step TRL PPO comparison", 45, weight="700")
    s.text(M, 205, "same trainer path, less repeated preparation", 31, "#4b5563")
    b, r, sv = TRL["baseline"], TRL["relpt"], TRL["savings"]
    max_ms = max(float(b["total_ms"]), float(r["total_ms"])) / 1000.0
    bar(s, M, 310, "total time", float(b["total_ms"]) / 1000.0, float(r["total_ms"]) / 1000.0, "s", max_ms)
    bar(s, M, 470, "prep time", float(b["prep_ms"]) / 1000.0, float(r["prep_ms"]) / 1000.0, "s", max_ms)
    bar(s, M, 630, "generate time", float(b["generate_ms"]) / 1000.0, float(r["generate_ms"]) / 1000.0, "s", max_ms)
    bar(s, M, 830, "generated rows", float(b["generated_rows"]), float(r["generated_rows"]), "", float(b["generated_rows"]))
    bar(s, M, 990, "reward rows", float(b["reward_rows"]), float(r["reward_rows"]), "", float(b["reward_rows"]))
    s.rect(M, 1225, W - 2 * M, 195, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1285, "Observed savings", 34, weight="700")
    s.text(M + 34, 1340, f"total {pct(sv['total_ms_saved_pct'])} | prep {pct(sv['prep_ms_saved_pct'])} | generated rows {pct(sv['generated_rows_saved_pct'])}", 29)
    s.text(M + 34, 1388, f"model sshleifer/tiny-gpt2 | dataset {TRL['config']['dataset_source']} | steps {TRL['config']['steps']}", 25, "#5e6978")
    s.footer(6)
    p = PAGES / "page_06.svg"
    s.save(p)
    return p


def page_07() -> Path:
    s = Svg("Full SQLite prototype result")
    s.text(M, 155, "Full RelPT prototype: large reward-cache reuse", 44, weight="700")
    n, r, sv = LOCAL["naive"], LOCAL["relpt"], LOCAL["savings"]
    max_rows = float(n["reward_rows"])
    bar(s, M, 310, "rollout rows", float(n["rollout_rows"]), float(r["rollout_rows"]), "", max_rows)
    bar(s, M, 470, "reward executor rows", float(n["reward_rows"]), float(r["reward_rows"]), "", max_rows)
    bar(s, M, 630, "logprob rows", float(n["logprob_rows"]), float(r["logprob_rows"]), "", max_rows)
    bar(s, M, 790, "train rows", float(n["train_rows"]), float(r["train_rows"]), "", max_rows)
    s.rect(M, 1060, W - 2 * M, 330, "#ffffff", "#d4dae3")
    s.text(M + 34, 1120, "Mechanism", 34, weight="700")
    s.multiline(
        M + 34,
        1180,
        f"At 200 steps, RelPT saves {pct(sv['rollout_rows_saved_pct'])} of rollout rows, "
        f"{pct(sv['reward_rows_saved_pct'])} of reward executor rows, and "
        f"{pct(sv['logprob_rows_saved_pct'])} of logprob rows. Train rows stay unchanged "
        "because PPO itself is intentionally the same black-box update.",
        30,
        56,
        43,
    )
    s.footer(7)
    p = PAGES / "page_07.svg"
    s.save(p)
    return p


def page_08() -> Path:
    s = Svg("Interpretation")
    s.text(M, 155, "How to read the evidence", 47, weight="700")
    cards = [
        ("Claim supported", "RelPT reduces duplicated PPO preparation work by materializing and reusing artifacts."),
        ("Claim not made", "The tiny model did not learn GSM8K math in this run; quality deltas are zero in the TRL job."),
        ("Why it matters", "Real post-training pipelines pay heavily for rollout, reward, and logprob executors."),
        ("Prime_TTT link", "The same relational planning idea fits the broader Prime_TTT view: optimize artifacts before the gradient sink."),
    ]
    y = 270
    for title, body in cards:
        s.rect(M, y, W - 2 * M, 210, "#ffffff", "#d4dae3")
        s.text(M + 34, y + 62, title, 32, weight="700")
        s.multiline(M + 34, y + 118, body, 29, 58, 42, "#394150")
        y += 250
    s.rect(M, 1320, W - 2 * M, 120, "#f6fff8", "#92c9a2")
    s.text(M + 34, 1375, "Shortest conclusion: RelPT is a relational reuse layer for post-training systems.", 30, weight="700")
    s.footer(8)
    p = PAGES / "page_08.svg"
    s.save(p)
    return p


def write_markdown() -> None:
    b, r, sv = TRL["baseline"], TRL["relpt"], TRL["savings"]
    local_sv = LOCAL["savings"]
    text = f"""# Prime_TTT RelPT Detailed Report

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

- model: `{TRL['config']['model_name']}`
- dataset: `{TRL['config']['dataset_source']}`
- framework: `{TRL['config']['framework']}`
- steps: `{TRL['config']['steps']}`
- batch size: `{TRL['config']['batch_size']}`
- output: `{TRL['config']['output']}`

| metric | baseline | relpt | saved |
| --- | ---: | ---: | ---: |
| total_ms | {float(b['total_ms']):.1f} | {float(r['total_ms']):.1f} | {pct(sv['total_ms_saved_pct'])} |
| prep_ms | {float(b['prep_ms']):.1f} | {float(r['prep_ms']):.1f} | {pct(sv['prep_ms_saved_pct'])} |
| generate_ms | {float(b['generate_ms']):.1f} | {float(r['generate_ms']):.1f} | {pct(sv['generate_ms_saved_pct'])} |
| reward_ms | {float(b['reward_ms']):.1f} | {float(r['reward_ms']):.1f} | {pct(sv['reward_ms_saved_pct'])} |
| generated_rows | {int(float(b['generated_rows']))} | {int(float(r['generated_rows']))} | {pct(sv['generated_rows_saved_pct'])} |
| reward_rows | {int(float(b['reward_rows']))} | {int(float(r['reward_rows']))} | {pct(sv['reward_rows_saved_pct'])} |

![TRL PPO result](figures/trl_ppo_result.svg)

## Complete SQLite RelPT prototype result

The full SQLite prototype saved {pct(local_sv['rollout_rows_saved_pct'])} of
rollout rows, {pct(local_sv['reward_rows_saved_pct'])} of reward executor rows,
and {pct(local_sv['logprob_rows_saved_pct'])} of logprob rows at 200 steps.
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
"""
    (DOCS / "relpt_report.md").write_text(text, encoding="utf-8")


def write_html() -> None:
    md = (DOCS / "relpt_report.md").read_text(encoding="utf-8")
    body: List[str] = []
    para: List[str] = []
    in_table = False
    in_ul = False
    in_ol = False

    def flush_para() -> None:
        if para:
            body.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def close_blocks() -> None:
        nonlocal in_table, in_ul, in_ol
        flush_para()
        if in_ul:
            body.append("</ul>")
            in_ul = False
        if in_ol:
            body.append("</ol>")
            in_ol = False
        if in_table:
            body.append("</tbody></table>")
            in_table = False

    for line in md.splitlines():
        if line.startswith("# "):
            close_blocks()
            body.append(f"<h1>{esc(line[2:])}</h1>")
        elif line.startswith("## "):
            close_blocks()
            body.append(f"<h2>{esc(line[3:])}</h2>")
        elif line.startswith("!["):
            close_blocks()
            alt = line.split("](", 1)[0][2:]
            src = line.split("](", 1)[1][:-1]
            body.append(f'<p><img src="{esc(src)}" alt="{esc(alt)}"></p>')
        elif line.startswith("- "):
            flush_para()
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{inline(line[2:])}</li>")
        elif len(line) > 2 and line[0].isdigit() and line[1:3] == ". ":
            flush_para()
            if not in_ol:
                body.append("<ol>")
                in_ol = True
            body.append(f"<li>{inline(line[3:])}</li>")
        elif line.startswith("| "):
            flush_para()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(cells[0]) <= {"-", ":"}:
                continue
            if not in_table:
                body.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
        elif not line.strip():
            close_blocks()
        else:
            para.append(line)
    close_blocks()
    html_text = f"""<!doctype html>
<meta charset="utf-8">
<title>Prime_TTT RelPT Detailed Report</title>
<style>
body {{ font-family: DejaVu Sans, Arial, sans-serif; max-width: 980px; margin: 40px auto; line-height: 1.65; color: #1f2933; }}
img {{ max-width: 100%; border: 1px solid #d6dce5; }}
code {{ background: #f3f6fa; padding: 2px 5px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
th, td {{ border: 1px solid #d6dce5; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f3f6fa; }}
td:nth-child(n+2), th:nth-child(n+2) {{ text-align: right; }}
table:nth-of-type(1) td:nth-child(2), table:nth-of-type(1) th:nth-child(2) {{ text-align: left; }}
</style>
<body>
{chr(10).join(body)}
</body>
"""
    (DOCS / "relpt_report.html").write_text(html_text, encoding="utf-8")


def inline(text: str) -> str:
    parts = text.split("`")
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(f"<code>{esc(part)}</code>")
        else:
            out.append(esc(part))
    return "".join(out)


def build_pdf(pages: Sequence[Path]) -> None:
    exe = shutil.which("rsvg-convert")
    if not exe:
        raise SystemExit("rsvg-convert is required to build docs/relpt_report.pdf")
    subprocess.run([exe, "-f", "pdf", "-o", str(DOCS / "relpt_report.pdf"), *map(str, pages)], check=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)
    pages = [fn() for fn in [page_01, page_02, page_03, page_04, page_05, page_06, page_07, page_08]]
    shutil.copyfile(pages[2], FIGURES / "relpt_pipeline.svg")
    shutil.copyfile(pages[5], FIGURES / "trl_ppo_result.svg")
    shutil.copyfile(pages[6], FIGURES / "simulated_relpt_result.svg")
    write_markdown()
    write_html()
    build_pdf(pages)
    print("Generated docs/relpt_report.md, .html, .pdf, figures, and report_pages")


if __name__ == "__main__":
    main()
