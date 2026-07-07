#!/usr/bin/env python3
"""Generate a Chinese RelPT explanation report with SVG figures and a PDF.

The environment intentionally avoids external Python packages.  This script
therefore writes SVG pages directly and asks rsvg-convert to assemble the PDF.
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
FONT_FAMILY = "Droid Sans Fallback, DejaVu Sans, Arial, sans-serif"
W, H = 1240, 1754
M = 72


def load_json(path: str) -> Dict[str, object]:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


TRL = load_json("runs/metrics/trl_ppo_relpt_17266929.json")
LOCAL = load_json("runs/metrics/relpt_gsm8k_200step_local.json")


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def wrap_text(text: str, chars: int) -> List[str]:
    lines: List[str] = []
    for para in str(text).split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        # Chinese has no spaces; textwrap still works acceptably by width.
        lines.extend(textwrap.wrap(para, width=chars, break_long_words=True, replace_whitespace=False))
    return lines


class Svg:
    def __init__(self, title: str):
        self.parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            '<rect width="100%" height="100%" fill="#fbfaf7"/>',
            f'<style>text{{font-family:{FONT_FAMILY};fill:#1f2933}} .muted{{fill:#5e6978}} .small{{font-size:23px}} .body{{font-size:28px}} .h1{{font-size:54px;font-weight:700}} .h2{{font-size:38px;font-weight:700}} .h3{{font-size:30px;font-weight:700}} .mono{{font-family:Source Code Pro,DejaVu Sans Mono,monospace}}</style>',
        ]
        self.text(M, 62, title, 23, "#697386")
        self.line(M, 82, W - M, 82, "#d7dbe2", 2)

    def text(self, x: float, y: float, value: object, size: int = 28, fill: str = "#1f2933", weight: str = "400", anchor: str = "start", cls: str = "") -> None:
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" class="{cls}">{esc(value)}</text>'
        )

    def multiline(self, x: float, y: float, text: str, size: int = 28, width_chars: int = 45, line_gap: int = 38, fill: str = "#1f2933") -> float:
        yy = y
        for line in wrap_text(text, width_chars):
            if line == "":
                yy += line_gap // 2
            else:
                self.text(x, yy, line, size, fill)
                yy += line_gap
        return yy

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "#ffffff", stroke: str = "#d9dee7", rx: int = 8, sw: int = 2) -> None:
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#9098a5", sw: int = 2, dash: str = "") -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#596579") -> None:
        self.line(x1, y1, x2, y2, stroke, 3)
        if abs(x2 - x1) >= abs(y2 - y1):
            sign = 1 if x2 > x1 else -1
            self.parts.append(f'<path d="M{x2},{y2} l{-16*sign},-9 l0,18 z" fill="{stroke}"/>')
        else:
            sign = 1 if y2 > y1 else -1
            self.parts.append(f'<path d="M{x2},{y2} l-9,{-16*sign} l18,0 z" fill="{stroke}"/>')

    def pill(self, x: float, y: float, text: str, fill: str, stroke: str = "#ccd3dd") -> None:
        self.rect(x, y, 254, 54, fill, stroke, 27, 1)
        self.text(x + 127, y + 36, text, 24, "#1f2933", "700", "middle")

    def footer(self, page: int) -> None:
        self.line(M, H - 74, W - M, H - 74, "#d7dbe2", 2)
        self.text(M, H - 35, "RelPT report generated from this repository", 21, "#697386")
        self.text(W - M, H - 35, f"{page}", 21, "#697386", anchor="end")

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def bar(svg: Svg, x: float, y: float, label: str, base: float, relpt: float, unit: str, maxv: float) -> None:
    svg.text(x, y + 30, label, 26, "#1f2933", "700")
    bx = x + 260
    bw = 610
    svg.rect(bx, y, bw, 34, "#edf1f5", "#edf1f5", 4, 0)
    svg.rect(bx, y, bw * base / maxv, 34, "#d86c59", "#d86c59", 4, 0)
    svg.text(bx + bw + 22, y + 27, f"baseline {base:.1f}{unit}", 23, "#5e6978")
    svg.rect(bx, y + 48, bw, 34, "#edf1f5", "#edf1f5", 4, 0)
    svg.rect(bx, y + 48, bw * relpt / maxv, 34, "#2f8f83", "#2f8f83", 4, 0)
    svg.text(bx + bw + 22, y + 75, f"RelPT {relpt:.1f}{unit}", 23, "#5e6978")


def page1() -> Path:
    s = Svg("What is RelPT?")
    s.text(M, 170, "RelPT 是什么，为什么这次实验要做它", 52, weight="700")
    y = s.multiline(
        M,
        245,
        "RelPT 不是新的 PPO loss，也不是新的大模型。它是 post-training 前处理阶段的关系型控制层：把 prompt、trajectory、reward、logprob、advantage、batch 等中间产物放进 append-only tables，用 SQL/join/cache 判断哪些东西已经算过，哪些东西还缺，然后只调用缺失部分的昂贵 executor。",
        31,
        43,
        44,
    )
    y += 28
    s.rect(M, y, W - 2 * M, 330, "#ffffff", "#d4dae3")
    s.text(M + 34, y + 62, "一句话版本", 34, weight="700")
    s.multiline(M + 34, y + 120, "Baseline 每次重新准备 PPO batch；RelPT 先查表，能复用 rollout/reward/logprob/batch 的就复用，只补缺口。PPO trainer 本身保持不变。", 32, 38, 46)
    y += 400
    s.text(M, y, "本报告覆盖", 37, weight="700")
    items = [
        "1. PPO baseline 从 prompt 到 update 的完整流程",
        "2. RelPT 加在哪一层，和 PPO/TRL/vLLM/reward model 的边界",
        "3. RelPT 使用哪些表、哪些 join、什么 SQL optimizer",
        "4. 当前仓库中两套实现：完整 SQLite 原型 + TRL PPO cache 版本",
        "5. 最新 200-step tiny-gpt2/GSM8K 结果图，以及结果应该如何解读",
    ]
    yy = y + 62
    for item in items:
        s.text(M + 20, yy, item, 29)
        yy += 50
    s.footer(1)
    p = PAGES / "page_01.svg"
    s.save(p)
    return p


def page2() -> Path:
    s = Svg("Baseline PPO dataflow")
    s.text(M, 160, "没有 RelPT 时：PPO batch preparation 每次重跑", 45, weight="700")
    boxes = [
        (M, 285, "GSM8K prompts", "问题文本 + 标准答案"),
        (M + 285, 285, "Policy model", "sshleifer/tiny-gpt2"),
        (M + 570, 285, "Rollout", "generate response"),
        (M + 855, 285, "Reward", "exact-answer scorer"),
        (M + 570, 505, "Logprob", "old policy logprob"),
        (M + 855, 505, "PPO batch", "queries/responses/scores"),
        (M + 855, 725, "TRL PPOTrainer", "same update path"),
    ]
    for x, y, title, sub in boxes:
        s.rect(x, y, 240, 112, "#ffffff", "#cbd3df")
        s.text(x + 120, y + 45, title, 25, weight="700", anchor="middle")
        s.text(x + 120, y + 82, sub, 21, "#697386", anchor="middle")
    s.arrow(M + 240, 341, M + 285, 341)
    s.arrow(M + 525, 341, M + 570, 341)
    s.arrow(M + 810, 341, M + 855, 341)
    s.arrow(M + 690, 397, M + 690, 505)
    s.arrow(M + 975, 397, M + 975, 505)
    s.arrow(M + 975, 617, M + 975, 725)
    s.rect(M, 1030, W - 2 * M, 330, "#fff7ed", "#efc58d")
    s.text(M + 34, 1090, "baseline 的问题", 34, weight="700")
    s.multiline(
        M + 34,
        1150,
        "如果同一个 policy step 的同一批 prompt 因为 retry、失败恢复、调参、K 从 4 增加到 6、或 reward/logprob 补算而再次准备 batch，baseline 往往重新 generate、重新 reward、重新 logprob。对于真实 LLM，generate 和 reward model 调用很贵。",
        30,
        48,
        43,
    )
    s.footer(2)
    p = PAGES / "page_02.svg"
    s.save(p)
    return p


def page3() -> Path:
    s = Svg("RelPT relational control layer")
    s.text(M, 155, "加上 RelPT 后：先查关系表，只补缺失行", 45, weight="700")
    xs = [M, M + 285, M + 570, M + 855]
    y = 270
    for i, name in enumerate(["prompt", "trajectory", "reward", "logprob"]):
        s.rect(xs[i], y, 240, 94, "#f7fbff", "#9db7d5")
        s.text(xs[i] + 120, y + 58, name, 28, weight="700", anchor="middle")
        if i:
            s.arrow(xs[i - 1] + 240, y + 47, xs[i], y + 47, "#5b7799")
    s.rect(M + 285, y + 190, 240, 94, "#f6fff8", "#92c9a2")
    s.text(M + 405, y + 248, "reward_cache", 26, weight="700", anchor="middle")
    s.arrow(M + 405, y + 190, M + 690, y + 94, "#5f9b6a")
    s.rect(M + 570, y + 190, 240, 94, "#f8f5ff", "#b6a4df")
    s.text(M + 690, y + 248, "advantage", 26, weight="700", anchor="middle")
    s.arrow(M + 690, y + 190, M + 690, y + 94, "#8068b5")
    s.rect(M + 855, y + 190, 240, 94, "#fffaf0", "#d9b36c")
    s.text(M + 975, y + 248, "batch", 26, weight="700", anchor="middle")
    s.arrow(M + 810, y + 47, M + 975, y + 190, "#9f7f39")
    s.text(M, 690, "RelPT optimizer 做的事情", 38, weight="700")
    steps = [
        ("1", "LEFT JOIN prompt/trajectory", "找当前 policy 下缺多少 rollout"),
        ("2", "JOIN trajectory/reward_cache", "同一 prompt + 同一 response hash 的 reward 直接复用"),
        ("3", "LEFT JOIN trajectory/logprob", "只对缺 logprob 的 trajectory 调 executor"),
        ("4", "JOIN 完整 rows", "materialize 成 PPO batch 交给 trainer"),
    ]
    yy = 760
    for n, title, desc in steps:
        s.pill(M, yy - 36, n, "#e8eef7")
        s.text(M + 300, yy, title, 30, weight="700")
        s.text(M + 300, yy + 42, desc, 27, "#5e6978")
        yy += 128
    s.footer(3)
    p = PAGES / "page_03.svg"
    s.save(p)
    return p


def page4() -> Path:
    s = Svg("Tables and join semantics")
    s.text(M, 155, "RelPT 的表和关系：不是 magic，是 SQL materialization", 43, weight="700")
    rows = [
        ("prompt", "输入 prompt、GSM8K 问题、标签/metadata"),
        ("policy", "PPO 更新前后的 policy version；每次 update append 一个版本"),
        ("trajectory", "rollout 结果：prompt_id、policy_id、response、response_hash"),
        ("reward", "某个 reward_model 对 trajectory 的 reward"),
        ("reward_cache", "按 reward_model_id + prompt_id + response_hash 复用确定性 reward"),
        ("logprob", "PPO 需要的 old-policy logprob"),
        ("advantage", "return / advantage，中间训练量"),
        ("batch", "materialized PPO batch membership"),
        ("update_record", "PPO 输入 policy、输出 policy、batch、trainer backend lineage"),
        ("metric", "训练和系统指标"),
    ]
    y = 235
    for i, (name, desc) in enumerate(rows):
        fill = "#ffffff" if i % 2 == 0 else "#f3f6fa"
        s.rect(M, y, 250, 58, fill, "#d6dce5", 2, 1)
        s.rect(M + 250, y, W - 2 * M - 250, 58, fill, "#d6dce5", 2, 1)
        s.text(M + 22, y + 38, name, 25, weight="700", cls="mono")
        s.text(M + 280, y + 38, desc, 25)
        y += 58
    s.rect(M, 930, W - 2 * M, 445, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 990, "SQL optimizer 是什么？", 34, weight="700")
    s.multiline(
        M + 34,
        1050,
        "当前实现用 Python 标准库 sqlite3。SQLite 负责真实 SQL 执行、索引/unique constraint 和本地 query planning。RelPT 自己的 optimizer 不是替代 SQLite，而是在 prepare_ppo_batch 前生成 worklists：rollout_requests、reward_traj_ids、logprob_traj_ids、advantage_traj_ids。然后 SQLite join 把已有 rows 和缺失 rows 分开。",
        29,
        51,
        42,
    )
    s.multiline(M + 34, 1262, "为什么这么做：因为 post-training 的昂贵部分通常在表外 executor，例如 LLM generation、reward model、logprob engine。SQL 层的任务是避免重复调用这些 executor。", 29, 51, 42)
    s.footer(4)
    p = PAGES / "page_04.svg"
    s.save(p)
    return p


def page5() -> Path:
    s = Svg("Implementation in this repository")
    s.text(M, 155, "这个仓库里的具体实现", 48, weight="700")
    s.rect(M, 245, W - 2 * M, 365, "#ffffff", "#cbd3df")
    s.text(M + 34, 305, "完整 RelPT SQLite 原型", 35, weight="700")
    s.multiline(M + 34, 365, "文件：relpt.py。核心类 RelPT 创建 append-only SQLite tables。RelPT.plan_ppo_batch 生成缺失 worklist；RelPT.prepare_ppo_batch 调用 rollout/reward/logprob/trainer adapters。测试 test_relpt.py 验证同一个 PPO batch retry 不再调用昂贵 executor。", 29, 49, 42)
    s.rect(M, 680, W - 2 * M, 365, "#ffffff", "#cbd3df")
    s.text(M + 34, 740, "GSM8K simulated benchmark", 35, weight="700")
    s.multiline(M + 34, 800, "文件：relpt_gsm8k_ppo.py。prompt 来自本地 GSM8K Arrow cache；rollout/PPO executor 是 toy softmax policy 的模拟器。它验证完整 RelPT 表结构能省 rollout/reward/logprob 行数，但不是实模型质量实验。", 29, 49, 42)
    s.rect(M, 1115, W - 2 * M, 365, "#ffffff", "#cbd3df")
    s.text(M + 34, 1175, "真实 TRL PPO same-framework 对照", 35, weight="700")
    s.multiline(M + 34, 1235, "文件：trl_gsm8k_relpt_ppo.py。baseline 和 RelPT 都调用同一个 TRL PPOTrainer，同一个模型 sshleifer/tiny-gpt2，同一个 GSM8K local Arrow cache。这里的 RelPT 是较小的 prepared-batch cache，目的是证明不改变 PPO update path 时也能减少重复 batch preparation。", 29, 49, 42)
    s.footer(5)
    p = PAGES / "page_05.svg"
    s.save(p)
    return p


def page6() -> Path:
    s = Svg("Latest TRL PPO result")
    s.text(M, 155, "最新 200-step TRL PPO：RelPT 前后差异", 45, weight="700")
    cfg = TRL["config"]  # type: ignore[index]
    s.rect(M, 225, W - 2 * M, 275, "#f8fbff", "#c8d7ea")
    s.text(M + 34, 285, "实验配置", 34, weight="700")
    lines = [
        f"model = {cfg['model_name']}",
        f"dataset = {cfg['dataset_source']}",
        f"framework = {cfg['framework']}",
        f"steps = {cfg['steps']}, batch_size = {cfg['batch_size']}, max_new_tokens = {cfg['max_new_tokens']}",
        f"job output = {cfg['output']}",
    ]
    yy = 335
    for line in lines:
        s.text(M + 48, yy, line, 25, "#394150", cls="mono")
        yy += 34
    b = TRL["baseline"]  # type: ignore[index]
    r = TRL["relpt"]  # type: ignore[index]
    max_time = max(float(b["total_ms"]), float(b["prep_ms"]), float(b["generate_ms"])) / 1000.0
    y = 585
    bar(s, M, y, "total time", float(b["total_ms"]) / 1000, float(r["total_ms"]) / 1000, "s", max_time)
    bar(s, M, y + 145, "prep time", float(b["prep_ms"]) / 1000, float(r["prep_ms"]) / 1000, "s", max_time)
    bar(s, M, y + 290, "generate time", float(b["generate_ms"]) / 1000, float(r["generate_ms"]) / 1000, "s", max_time)
    max_rows = max(float(b["generated_rows"]), float(b["reward_rows"]))
    bar(s, M, y + 470, "generated rows", float(b["generated_rows"]), float(r["generated_rows"]), "", max_rows)
    bar(s, M, y + 615, "reward rows", float(b["reward_rows"]), float(r["reward_rows"]), "", max_rows)
    s.rect(M, 1480, W - 2 * M, 150, "#ffffff", "#d6dce5")
    savings = TRL["savings"]  # type: ignore[index]
    s.text(M + 34, 1530, f"总时间节省 {float(savings['total_ms_saved_pct']):.1f}%；生成行数和 reward 行数各减少 50.0%。", 29, weight="700")
    s.multiline(
        M + 34,
        1572,
        "final_eval_reward_delta = 0.0，说明这次实验主要证明系统省计算；它不证明 tiny-gpt2 数学能力提升。",
        24,
        58,
        32,
        "#5e6978",
    )
    s.footer(6)
    p = PAGES / "page_06.svg"
    s.save(p)
    return p


def page7() -> Path:
    s = Svg("Full RelPT simulated result")
    s.text(M, 155, "完整 RelPT 表结构的 200-step 模拟结果", 44, weight="700")
    n = LOCAL["naive"]  # type: ignore[index]
    r = LOCAL["relpt"]  # type: ignore[index]
    savings = LOCAL["savings"]  # type: ignore[index]
    max_rows = max(float(n["rollout_rows"]), float(n["reward_rows"]), float(n["logprob_rows"]))
    y = 280
    bar(s, M, y, "rollout rows", float(n["rollout_rows"]), float(r["rollout_rows"]), "", max_rows)
    bar(s, M, y + 150, "reward rows", float(n["reward_rows"]), float(r["reward_rows"]), "", max_rows)
    bar(s, M, y + 300, "logprob rows", float(n["logprob_rows"]), float(r["logprob_rows"]), "", max_rows)
    bar(s, M, y + 450, "train rows", float(n["train_rows"]), float(r["train_rows"]), "", max_rows)
    s.rect(M, 1015, W - 2 * M, 410, "#fffdfa", "#dfc89f")
    s.text(M + 34, 1075, "这张图的含义", 34, weight="700")
    s.multiline(
        M + 34,
        1135,
        f"在完整 SQLite RelPT 原型里，rollout_rows 省 {float(savings['rollout_rows_saved_pct']):.1f}%，logprob_rows 省 {float(savings['logprob_rows_saved_pct']):.1f}%，reward_rows 省 {float(savings['reward_rows_saved_pct']):.1f}%。train_rows 省 0.0%，这是故意的：RelPT 不改 PPO 更新本身，只优化 PPO 前后的数据准备和复用。",
        30,
        48,
        43,
    )
    s.multiline(M + 34, 1320, "reward_rows 省得最多，是因为 reward_cache 可以跨 policy version 复用 deterministic exact-answer reward。", 30, 48, 43)
    s.footer(7)
    p = PAGES / "page_07.svg"
    s.save(p)
    return p


def page8() -> Path:
    s = Svg("Conclusion")
    s.text(M, 155, "从开始到结束：应该怎样理解 RelPT", 47, weight="700")
    conclusions = [
        ("PPO on what model?", "最新真实 TRL 对照使用 sshleifer/tiny-gpt2，在本地 GSM8K Arrow cache 上跑 200 steps。"),
        ("加 RelPT 前", "每次 batch preparation 都重新 generate/reward/logprob；retry 或重复准备会浪费昂贵 executor 调用。"),
        ("加 RelPT 后", "先查关系表/cache。已有 trajectory/reward/logprob/batch 直接 materialize；缺什么补什么。"),
        ("SQL optimizer", "SQLite 是底层 SQL engine。RelPT 的 delta planner 负责生成缺失 worklist，SQLite join/constraint 负责复用和去重。"),
        ("为什么这么做", "真实 post-training 的成本通常不在 SQL，而在 LLM generation、reward model 和 logprob engine。把中间产物表化后，就能避免重复调用这些黑盒 executor。"),
        ("结果怎么读", "最新 TRL 200-step 总时间省 40.7%，generation/reward rows 各省 50%。质量 delta 为 0，说明 PPO 更新路径没变；这次主要是系统效率证据。"),
    ]
    y = 260
    for title, body in conclusions:
        s.rect(M, y, W - 2 * M, 145, "#ffffff", "#d6dce5")
        s.text(M + 34, y + 50, title, 30, weight="700")
        s.multiline(M + 285, y + 44, body, 27, 38, 38, "#394150")
        y += 175
    s.rect(M, 1390, W - 2 * M, 130, "#eef7f4", "#a7d1c1")
    s.multiline(M + 34, 1445, "最短结论：RelPT 是 post-training pipeline 的关系型控制层。它不声称 PPO loss 更好，而是让同一个 PPO trainer 少重复准备数据。", 31, 48, 44)
    s.footer(8)
    p = PAGES / "page_08.svg"
    s.save(p)
    return p


def write_figure_files() -> None:
    # Standalone SVGs for GitHub markdown preview.
    for name, page_fn in [
        ("relpt_pipeline.svg", page3),
        ("trl_ppo_result.svg", page6),
        ("simulated_relpt_result.svg", page7),
    ]:
        tmp = page_fn()
        (FIGURES / name).write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")


def write_markdown() -> None:
    cfg = TRL["config"]  # type: ignore[index]
    b = TRL["baseline"]  # type: ignore[index]
    r = TRL["relpt"]  # type: ignore[index]
    s = TRL["savings"]  # type: ignore[index]
    md = f"""# RelPT 从零解释报告

这份报告解释本仓库里的 RelPT：它是什么、用哪些 table 和 SQL join、optimizer 做什么、为什么这样做，以及最新 PPO 对照结果怎么读。

## 一句话

RelPT 不是新的 PPO loss。它是 post-training 前处理阶段的关系型控制层：把 prompt、trajectory、reward、logprob、advantage、batch 等中间产物表化，然后只补缺失 rows，避免重复调用 rollout/reward/logprob executor。

## 图 1：RelPT 控制层

![RelPT pipeline](figures/relpt_pipeline.svg)

## 最新真实 TRL PPO 对照

- model: `{cfg['model_name']}`
- dataset: `{cfg['dataset_source']}`
- framework: `{cfg['framework']}`
- steps: `{cfg['steps']}`
- batch size: `{cfg['batch_size']}`
- output: `{cfg['output']}`

| metric | baseline | relpt | saved |
| --- | ---: | ---: | ---: |
| total_ms | {float(b['total_ms']):.1f} | {float(r['total_ms']):.1f} | {float(s['total_ms_saved_pct']):.1f}% |
| prep_ms | {float(b['prep_ms']):.1f} | {float(r['prep_ms']):.1f} | {float(s['prep_ms_saved_pct']):.1f}% |
| generate_ms | {float(b['generate_ms']):.1f} | {float(r['generate_ms']):.1f} | {float(s['generate_ms_saved_pct']):.1f}% |
| reward_ms | {float(b['reward_ms']):.1f} | {float(r['reward_ms']):.1f} | {float(s['reward_ms_saved_pct']):.1f}% |
| generated_rows | {float(b['generated_rows']):.0f} | {float(r['generated_rows']):.0f} | {float(s['generated_rows_saved_pct']):.1f}% |
| reward_rows | {float(b['reward_rows']):.0f} | {float(r['reward_rows']):.0f} | {float(s['reward_rows_saved_pct']):.1f}% |

![TRL PPO result](figures/trl_ppo_result.svg)

## 完整 SQLite RelPT 原型结果

![Simulated RelPT result](figures/simulated_relpt_result.svg)

## 怎么读结果

这次结果说明 RelPT 可以减少 PPO batch preparation 的重复工作。它没有说明 tiny-gpt2 在 GSM8K 上学会了数学，因为 final_eval_reward_delta 和 mean_train_reward_delta 都是 0.0。这个实验的重点是系统效率：同一个 PPOTrainer、同一个模型、同一个数据集，RelPT 减少了 generate/reward rows 和 preparation time。

PDF 版本见 `docs/relpt_report.pdf`。
"""
    (DOCS / "relpt_report.md").write_text(md, encoding="utf-8")


def write_html() -> None:
    md = (DOCS / "relpt_report.md").read_text(encoding="utf-8")
    html_body = "<br>\n".join(esc(line) for line in md.splitlines())
    html_body = html_body.replace("![RelPT pipeline](figures/relpt_pipeline.svg)", '<img src="figures/relpt_pipeline.svg">')
    html_body = html_body.replace("![TRL PPO result](figures/trl_ppo_result.svg)", '<img src="figures/trl_ppo_result.svg">')
    html_body = html_body.replace("![Simulated RelPT result](figures/simulated_relpt_result.svg)", '<img src="figures/simulated_relpt_result.svg">')
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>RelPT report</title>
<style>
body {{ font-family: {FONT_FAMILY}; max-width: 980px; margin: 40px auto; line-height: 1.65; color: #1f2933; }}
img {{ max-width: 100%; border: 1px solid #d6dce5; }}
code {{ background: #f3f6fa; padding: 2px 5px; }}
</style>
<body>{html_body}</body>
"""
    (DOCS / "relpt_report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)
    pages = [page1(), page2(), page3(), page4(), page5(), page6(), page7(), page8()]
    write_figure_files()
    # Regenerate pages after standalone figure writes, because page functions write report pages too.
    pages = [page1(), page2(), page3(), page4(), page5(), page6(), page7(), page8()]
    write_markdown()
    write_html()
    out_pdf = DOCS / "relpt_report.pdf"
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise SystemExit("rsvg-convert not found; SVG and Markdown were generated, PDF was not.")
    subprocess.run([converter, "-f", "pdf", "-o", str(out_pdf), *map(str, pages)], check=True)
    print(out_pdf)


if __name__ == "__main__":
    main()
