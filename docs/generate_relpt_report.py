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
    s = Svg("Optimization target")
    s.text(M, 155, "Target: minimize executor calls", 48, weight="700")
    s.text(M, 212, "while still producing the exact PPO batch relation", 32, "#4b5563")
    s.rect(M, 300, W - 2 * M, 250, "#ffffff", "#d4dae3")
    s.text(M + 34, 362, "RelPT planning equation", 34, weight="700")
    s.text(M + 34, 438, "required PPO rows - existing materialized rows = missing executor work", 31, "#1f2933")
    s.text(M + 34, 494, "The optimizer target is small missing work, not a different PPO objective.", 27, "#4b5563")
    nodes = [
        ("Required", "prompt x policy x K", "#f7fbff"),
        ("Existing", "artifact tables", "#f6fff8"),
        ("Missing", "worklists", "#fff7ed"),
        ("Trainer", "same PPO batch", "#f8f5ff"),
    ]
    x = M
    for i, (title, body, fill) in enumerate(nodes):
        node(s, x, 700, title, body, fill, 238, 130)
        if i:
            s.arrow(x - 47, 765, x, 765)
        x += 285
    s.rect(M, 1010, W - 2 * M, 345, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1072, "Concrete target example", 34, weight="700")
    s.multiline(
        M + 34,
        1132,
        "For 96 prompts, suppose the next PPO batch asks for K=6 responses per prompt, "
        "but the database already has K=4 usable trajectories for the same policy. "
        "The required relation has 576 trajectory rows. The existing relation has 384. "
        "The missing relation has 192 rows. Only those 192 rows should call rollout.",
        29,
        58,
        42,
    )
    s.footer(1)
    p = PAGES / "page_01.svg"
    s.save(p)
    return p


def page_02() -> Path:
    s = Svg("Why PPO is tables")
    s.text(M, 155, "A PPO batch is already a join of artifacts", 45, weight="700")
    s.text(M, 210, "One train row is valid only when all required artifacts exist.", 30, "#4b5563")
    node(s, M, 330, "prompt", "question text", "#f7fbff", 220, 120)
    node(s, M + 255, 330, "policy", "version p_t", "#f8f5ff", 220, 120)
    node(s, M + 510, 330, "trajectory", "response y", "#fff7ed", 220, 120)
    node(s, M + 765, 250, "reward", "score r", "#f6fff8", 220, 120)
    node(s, M + 765, 410, "logprob", "old log p", "#eef2ff", 220, 120)
    s.arrow(M + 220, 390, M + 255, 390)
    s.arrow(M + 475, 390, M + 510, 390)
    s.arrow(M + 730, 390, M + 765, 310)
    s.arrow(M + 730, 390, M + 765, 470)
    node(s, M + 370, 650, "advantage", "from reward/logprob", "#fffaf0", 240, 120)
    node(s, M + 710, 650, "PPO batch row", "query, response, r, logp, A", "#ffffff", 320, 120)
    s.arrow(M + 875, 530, M + 830, 650)
    s.arrow(M + 490, 450, M + 490, 650)
    s.arrow(M + 610, 710, M + 710, 710)
    s.rect(M, 920, W - 2 * M, 370, "#ffffff", "#d4dae3")
    s.text(M + 34, 982, "Relational formulation", 34, weight="700")
    lines = [
        "trajectory joins prompt on prompt_id and policy on policy_id",
        "reward joins trajectory on traj_id plus reward_model_id",
        "logprob joins trajectory on traj_id plus old_policy_id",
        "advantage joins reward/logprob for derived training credit",
        "batch is the materialized set of complete joined rows",
    ]
    y = 1048
    for line in lines:
        s.text(M + 34, y, "- " + line, 27)
        y += 52
    s.rect(M, 1370, W - 2 * M, 90, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1425, "SQL JOIN is not cosmetic; it is the PPO data contract made explicit.", 27, weight="700")
    s.footer(2)
    p = PAGES / "page_02.svg"
    s.save(p)
    return p


def page_03() -> Path:
    s = Svg("Anti-join planning")
    s.text(M, 155, "Anti-join planning finds missing rows", 45, weight="700")
    s.text(M, 210, "SQL asks: required rows LEFT JOIN existing rows, then keep NULL matches.", 29, "#4b5563")
    node(s, M, 330, "required", "prompt x policy x K", "#f7fbff", 260, 130)
    node(s, M + 385, 330, "existing", "trajectory", "#f6fff8", 260, 130)
    node(s, M + 770, 330, "missing", "rollout_requests", "#fff7ed", 260, 130)
    s.arrow(M + 260, 395, M + 385, 395)
    s.arrow(M + 645, 395, M + 770, 395)
    s.text(M + 255, 520, "LEFT JOIN ... WHERE existing.id IS NULL", 28, "#4b5563", cls="mono")
    s.rect(M, 660, W - 2 * M, 510, "#ffffff", "#d4dae3")
    s.text(M + 34, 722, "Repeated for every artifact layer", 34, weight="700")
    rows = [
        ("rollout", "prompt x policy x K", "trajectory", "rollout_requests"),
        ("reward", "trajectory x reward_model", "reward/cache", "reward_traj_ids"),
        ("logprob", "trajectory x old_policy", "logprob", "logprob_traj_ids"),
        ("advantage", "rewarded trajectory", "advantage", "advantage_traj_ids"),
        ("batch", "complete joined rows", "batch", "materialized batch"),
    ]
    y = 800
    for stage, need, have, out in rows:
        s.text(M + 34, y, stage, 25, weight="700", cls="mono")
        s.text(M + 190, y, need, 24, "#4b5563")
        s.text(M + 548, y, "-", 26, "#697386")
        s.text(M + 600, y, have, 24, "#4b5563")
        s.text(M + 805, y, "-> " + out, 24, "#1f2933")
        y += 75
    s.rect(M, 1270, W - 2 * M, 130, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1330, "DB role: physical joins, indexes, and materialization. RelPT supplies the artifact graph.", 25)
    s.footer(3)
    p = PAGES / "page_03.svg"
    s.save(p)
    return p


def page_04() -> Path:
    s = Svg("Partial rollout example")
    s.text(M, 155, "Example: partial rollout, K=4 to K=6", 43, weight="700")
    s.text(M, 210, "The target relation expands; existing rows stay valid.", 30, "#4b5563")
    s.rect(M, 310, W - 2 * M, 250, "#ffffff", "#d4dae3")
    s.text(M + 34, 372, "Per prompt", 34, weight="700")
    node(s, M + 70, 420, "existing", "4 trajectories", "#f6fff8", 260, 100)
    node(s, M + 420, 420, "target", "6 trajectories", "#f7fbff", 260, 100)
    node(s, M + 770, 420, "missing", "2 trajectories", "#fff7ed", 260, 100)
    s.arrow(M + 330, 470, M + 420, 470)
    s.arrow(M + 680, 470, M + 770, 470)
    s.rect(M, 700, 500, 330, "#fff7ed", "#efc58d")
    s.text(M + 34, 765, "Naive pipeline", 34, weight="700")
    s.text(M + 34, 835, "96 prompts x 6", 31)
    s.text(M + 34, 900, "= 576 rollout calls", 36, weight="700")
    s.text(M + 34, 960, "384 reusable rows are ignored", 25, "#697386")
    s.rect(M + 596, 700, 500, 330, "#f6fff8", "#92c9a2")
    s.text(M + 630, 765, "RelPT", 34, weight="700")
    s.text(M + 630, 835, "96 prompts x (6 - 4)", 31)
    s.text(M + 630, 900, "= 192 rollout calls", 36, weight="700")
    s.text(M + 630, 960, "only missing rows call rollout", 25, "#697386")
    s.rect(M, 1190, W - 2 * M, 210, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1250, "SQL sketch", 34, weight="700")
    s.text(M + 34, 1310, "GROUP BY prompt_id, policy_id; existing_k = count(trajectory)", 25, cls="mono")
    s.text(M + 34, 1360, "rollout_requests = target_k - existing_k where existing_k < target_k", 25, cls="mono")
    s.footer(4)
    p = PAGES / "page_04.svg"
    s.save(p)
    return p


def page_05() -> Path:
    s = Svg("Table interactions")
    s.text(M, 155, "Rows unlock downstream joins as tables grow", 44, weight="700")
    node(s, M, 305, "trajectory", "prompt_id, policy_id, response_hash", "#fff7ed", 300, 130)
    node(s, M + 420, 215, "reward_cache", "reward_model, prompt, hash", "#f6fff8", 300, 130)
    node(s, M + 420, 395, "reward", "traj_id, score", "#f6fff8", 300, 130)
    node(s, M + 830, 305, "logprob", "traj_id, old_policy", "#eef2ff", 300, 130)
    s.arrow(M + 300, 370, M + 420, 280, "#5f9b6a")
    s.arrow(M + 300, 370, M + 420, 460, "#5f9b6a")
    s.arrow(M + 720, 460, M + 830, 370, "#8068b5")
    node(s, M + 270, 660, "advantage", "reward-derived credit", "#fffaf0", 300, 120)
    node(s, M + 650, 660, "batch", "complete joined rows", "#f8f5ff", 300, 120)
    s.arrow(M + 570, 720, M + 650, 720)
    s.rect(M, 900, W - 2 * M, 420, "#ffffff", "#d4dae3")
    s.text(M + 34, 960, "Reuse rules are different for each table", 34, weight="700")
    rows = [
        ("trajectory", "reused for same prompt, same policy, same rollout slot or response row"),
        ("reward_cache", "reused across policy versions if prompt + normalized response hash match"),
        ("logprob", "usually policy-specific; missing old-policy logprob must be computed"),
        ("advantage", "derived from reward/logprob; recompute only when inputs or baseline change"),
        ("batch", "materialized complete join; exact retry can reuse with zero executor calls"),
    ]
    y = 1032
    for name, desc in rows:
        s.text(M + 34, y, name, 25, weight="700", cls="mono")
        s.text(M + 230, y, desc, 24, "#4b5563")
        y += 64
    s.rect(M, 1390, W - 2 * M, 90, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1445, "This is why the tables interact: a new upstream row changes which downstream joins are now complete.", 25)
    s.footer(5)
    p = PAGES / "page_05.svg"
    s.save(p)
    return p


def page_06() -> Path:
    s = Svg("Why DB optimization applies")
    s.text(M, 155, "Why a DB optimizer can help", 48, weight="700")
    cards = [
        ("Stable keys", "prompt_id, policy_id, traj_id, reward_model_id, response_hash"),
        ("Append-only facts", "executor outputs are durable rows, not hidden process state"),
        ("Completeness query", "PPO input is complete only after the artifact joins succeed"),
        ("Anti-join worklists", "missing rows are exactly what expensive executors must produce"),
        ("Cost asymmetry", "rollout/reward/logprob are costly; SQL joins and indexes are cheap"),
    ]
    y = 270
    for title, body in cards:
        s.rect(M, y, W - 2 * M, 165, "#ffffff", "#d4dae3")
        s.text(M + 34, y + 60, title, 32, weight="700")
        s.text(M + 330, y + 60, body, 27, "#394150")
        y += 198
    s.rect(M, 1310, W - 2 * M, 135, "#f6fff8", "#92c9a2")
    s.text(M + 34, 1370, "Important boundary", 32, weight="700")
    s.text(M + 330, 1370, "RelPT optimizes artifact construction, not PPO loss math.", 27)
    s.footer(6)
    p = PAGES / "page_06.svg"
    s.save(p)
    return p


def page_07() -> Path:
    s = Svg("Measured effects")
    s.text(M, 150, "Measured effects: less repeated work", 45, weight="700")
    s.text(M, 205, "Same trainer path; savings come from artifact reuse.", 30, "#4b5563")
    b, tr, trsv = TRL["baseline"], TRL["relpt"], TRL["savings"]
    max_ms = max(float(b["total_ms"]), float(tr["total_ms"])) / 1000.0
    bar(s, M, 290, "TRL total time", float(b["total_ms"]) / 1000.0, float(tr["total_ms"]) / 1000.0, "s", max_ms)
    bar(s, M, 450, "TRL prep time", float(b["prep_ms"]) / 1000.0, float(tr["prep_ms"]) / 1000.0, "s", max_ms)
    bar(s, M, 610, "TRL generated rows", float(b["generated_rows"]), float(tr["generated_rows"]), "", float(b["generated_rows"]))
    n, r, sv = LOCAL["naive"], LOCAL["relpt"], LOCAL["savings"]
    max_rows = float(n["reward_rows"])
    bar(s, M, 810, "SQLite rollout rows", float(n["rollout_rows"]), float(r["rollout_rows"]), "", max_rows)
    bar(s, M, 970, "SQLite reward calls", float(n["reward_rows"]), float(r["reward_rows"]), "", max_rows)
    s.rect(M, 1215, W - 2 * M, 220, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 1275, "Readout", 34, weight="700")
    s.multiline(
        M + 34,
        1335,
        f"TRL saved {pct(trsv['prep_ms_saved_pct'])} prep time and {pct(trsv['generated_rows_saved_pct'])} generated rows. "
        f"The full SQLite prototype saved {pct(sv['reward_rows_saved_pct'])} reward executor calls because reward_cache reused deterministic scores.",
        28,
        56,
        40,
    )
    s.footer(7)
    p = PAGES / "page_07.svg"
    s.save(p)
    return p


def page_08() -> Path:
    s = Svg("Interpretation")
    s.text(M, 155, "What the example proves", 47, weight="700")
    cards = [
        ("Target", "Produce the complete PPO batch relation with minimum additional executor calls."),
        ("Mechanism", "Represent artifacts as growing tables; use joins and anti-joins to find missing rows."),
        ("Examples", "Retry becomes zero new work. Partial rollout K=4 to K=6 becomes only the K delta. Deterministic rewards hit reward_cache."),
        ("Boundary", "RelPT does not claim a better PPO loss. It claims a better systems plan for constructing PPO inputs."),
    ]
    y = 270
    for title, body in cards:
        s.rect(M, y, W - 2 * M, 210, "#ffffff", "#d4dae3")
        s.text(M + 34, y + 62, title, 32, weight="700")
        s.multiline(M + 34, y + 118, body, 29, 58, 42, "#394150")
        y += 250
    s.rect(M, 1320, W - 2 * M, 120, "#f6fff8", "#92c9a2")
    s.text(M + 34, 1375, "Shortest conclusion: RelPT optimizes the dataflow before the gradient update.", 30, weight="700")
    s.footer(8)
    p = PAGES / "page_08.svg"
    s.save(p)
    return p


def write_markdown() -> None:
    b, r, sv = TRL["baseline"], TRL["relpt"], TRL["savings"]
    local_sv = LOCAL["savings"]
    text = f"""# Prime_TTT RelPT Detailed Report

RelPT is the relational planning layer in Prime_TTT. Its target is simple:
produce the complete PPO batch relation while minimizing new rollout, reward,
logprob, and advantage work.

## The target

RelPT does not change PPO loss. It changes how the input artifacts for PPO are
constructed.

`required PPO rows - existing materialized rows = missing executor work`

The optimizer should make the missing work small. The trainer should still see
the same query, response, reward, old-logprob, and advantage batch.

## Why PPO can be tables

A PPO batch row is not a single opaque object. It is a join of artifacts:

| table | example key | why PPO needs it |
| --- | --- | --- |
| `prompt` | `prompt_id` | query text and dataset metadata |
| `policy` | `policy_id` | which model version generated or scored the row |
| `trajectory` | `traj_id`, `prompt_id`, `policy_id` | generated response tokens/text |
| `reward` | `traj_id`, `reward_model_id` | score used by PPO |
| `logprob` | `traj_id`, `old_policy_id` | old-policy probability ratio reference |
| `advantage` | `traj_id`, `policy_id` | credit used by the PPO objective |
| `batch` | `batch_id`, `traj_id` | materialized membership consumed by PPOTrainer |

The final training relation is the set of rows where these joins are complete.
That is why SQL joins are natural here: PPO already requires consistency across
these artifacts. RelPT makes the consistency check explicit.

![RelPT pipeline](figures/relpt_pipeline.svg)

## How missing work is computed

RelPT uses anti-joins: build the required relation, left join existing artifact
tables, and keep the rows where the existing side is NULL.

| stage | required relation | existing table | missing output |
| --- | --- | --- | --- |
| rollout | `prompt x policy x target_K` | `trajectory` | `rollout_requests` |
| reward | `trajectory x reward_model` | `reward` / `reward_cache` | `reward_traj_ids` |
| logprob | `trajectory x old_policy` | `logprob` | `logprob_traj_ids` |
| advantage | rewarded trajectory rows | `advantage` | `advantage_traj_ids` |
| batch | complete joined rows | `batch` | materialized PPO batch |

The database optimizer matters because these are physical join/index/materialize
questions over growing tables. RelPT supplies the logical artifact graph; SQLite
executes the joins and constraints in this prototype.

## Concrete example: partial rollout

Suppose the target batch asks for `K=6` rollouts per prompt over 96 prompts, but
the database already has `K=4` usable trajectories for the same policy.

| plan | rollout calls |
| --- | ---: |
| naive | `96 * 6 = 576` |
| RelPT | `96 * (6 - 4) = 192` |

The SQL shape is:

`GROUP BY prompt_id, policy_id; existing_k = count(trajectory)`

`rollout_requests = target_k - existing_k where existing_k < target_k`

The same logic applies after rollout. If the new 192 trajectories do not have
rewards, the reward anti-join emits only those trajectory ids. If an old
trajectory already has a deterministic reward in `reward_cache`, no reward
executor call is needed. If logprobs are missing for the relevant old policy,
only those logprob rows are computed.

## Why this can work

It works when executor outputs can be represented as durable keyed artifacts:
rollout rows, reward rows, logprob rows, advantages, and batches. Once each
artifact has stable keys, the question "what should I compute next?" becomes a
database completeness query rather than ad hoc Python control flow.

This is also why different tables have different reuse rules:

- `trajectory` reuse is tied to prompt, policy, and rollout requirement.
- `reward_cache` can cross policy versions when prompt plus normalized response hash match.
- `logprob` is usually policy-specific, so it cannot be blindly reused across policies.
- `advantage` is derived from reward/logprob/baseline inputs.
- `batch` is a materialized complete join, so exact retries can reuse it directly.

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
    shutil.copyfile(pages[6], FIGURES / "trl_ppo_result.svg")
    shutil.copyfile(pages[4], FIGURES / "simulated_relpt_result.svg")
    write_markdown()
    write_html()
    build_pdf(pages)
    print("Generated docs/relpt_report.md, .html, .pdf, figures, and report_pages")


if __name__ == "__main__":
    main()
