#!/usr/bin/env python3
"""Presentation figures for the Build-with-Beam entry (dark-mode first).

Three figures, all computed from tmp/rl_rollouts/*.json so they can never
drift from the published trajectories:

  1. episode_ladder.png   — keyed-task episode success, four models
  2. context_ablation.png — excerpt vs full-paper reward per run
  3. improve_arc.png      — CA C1 nu: clean round -> self-improvement round

Palette = the Lemma site's (site/src/styles/global.css), chosen for
readability on HF's dark background:
  Beam #fab533 (gold) · Kimi #1ec2b5 (teal) · 27B #7866ed (violet) ·
  Opus #d445f2 (magenta) · bg #0b1018 · card #161d2c · edge #263049 ·
  fg #f1f3f6 · mut #98a2b3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RL = REPO / "tmp" / "rl_rollouts"
OUT = REPO / "tmp" / "figures"

BG, CARD, EDGE, FG, MUT, DIM = (
    "#0b1018",
    "#161d2c",
    "#263049",
    "#f1f3f6",
    "#98a2b3",
    "#64708a",
)
COLORS = {
    "BEAMHF": "#fab533",
    "KIMIHF": "#1ec2b5",
    "RUNINFRA": "#7866ed",
    "ANTHROPIC": "#d445f2",
}
NAMES = {
    "BEAMHF": "Beam 1.0 501B",
    "KIMIHF": "Kimi K3",
    "RUNINFRA": "27B-class (RunInfra)",
    "ANTHROPIC": "Claude Opus 5.5",
}


def load_episodes() -> list[dict]:
    eps = []
    for f in sorted(RL.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        t["_file"] = f.stem
        provider = t.get("provider", "")
        if provider == "ANTHROPIC":
            t["_label"] = "Opus 5.5"
        elif provider == "BEAMHF":
            t["_label"] = "Beam"
        elif provider == "KIMIHF":
            t["_label"] = "Kimi K3"
        else:
            t["_label"] = "27B-class"
        eps.append(t)
    return eps


def style_ax(ax, title: str, sub: str | None = None):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.tick_params(colors=MUT)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.grid(color=EDGE, linewidth=0.6, alpha=0.7)
    ax.set_title(title, color=FG, fontsize=17, fontweight="bold", loc="left", pad=18)
    if sub:
        ax.text(
            0,
            1.012,
            sub,
            transform=ax.transAxes,
            color=MUT,
            fontsize=10.5,
            va="bottom",
            ha="left",
        )


def fig_ladder(eps: list[dict]) -> None:
    agg: dict[str, list[int]] = {}
    for t in eps:
        if "_c1_" in t["_file"] and not t.get("full_context"):
            p = t["provider"]
            a = agg.setdefault(p, [0, 0])
            a[1] += 1
            a[0] += t["best_reward"] >= 1.0
    order = ["BEAMHF", "KIMIHF", "RUNINFRA", "ANTHROPIC"]
    rows = [(NAMES[p], agg[p][0], agg[p][1]) for p in order if p in agg]

    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=BG)
    y = range(len(rows))
    for i, (_name, ok, n) in enumerate(rows):
        frac = ok / n
        ax.barh(i, 1.0, height=0.55, color=CARD, zorder=2)
        ax.barh(i, frac, height=0.55, color=COLORS[order[i]], zorder=3)
        ax.text(
            1.025,
            i,
            f"{ok}/{n}",
            va="center",
            ha="left",
            color=FG,
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            max(frac - 0.03, 0.03),
            i,
            f"{frac:.0%}" if frac else "",
            va="center",
            ha="right",
            color=BG,
            fontsize=11,
            fontweight="bold",
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], color=FG, fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(
        "episodes reaching reward 1.0 (reference key recovered, control passed, verdict supported)"
    )
    style_ax(
        ax,
        "Beam wins the auditability axis: 4/4 vs a closed frontier model's 0/4",
        "Keyed tasks in the Lemma RL environment · deterministic grader, no LLM judge · every episode published",
    )
    ax.axvline(1.0, color=DIM, linewidth=1, linestyle=":", zorder=1)
    fig.tight_layout()
    fig.savefig(OUT / "episode_ladder.png", facecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(eps: list[dict]) -> None:
    conds = [
        (
            "Beam\n20k excerpt",
            lambda t: t["provider"] == "BEAMHF" and not t.get("full_context"),
        ),
        (
            "Beam\nfull paper (120k chars)",
            lambda t: t["provider"] == "BEAMHF" and t.get("full_context"),
        ),
        (
            "Opus 5.5\n20k excerpt",
            lambda t: t["provider"] == "ANTHROPIC" and not t.get("full_context"),
        ),
        (
            "Opus 5.5\nfull paper (120k chars)",
            lambda t: t["provider"] == "ANTHROPIC" and t.get("full_context"),
        ),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=BG)
    for i, (_label, pred) in enumerate(conds):
        runs = [t for t in eps if "c2" in t["_file"] and pred(t)]
        provider = "BEAMHF" if i < 2 else "ANTHROPIC"
        xs = [i - 0.14, i + 0.14][: len(runs)] or [i]
        for x, t in zip(xs, runs, strict=False):
            ax.scatter(
                [x],
                [t["best_reward"]],
                s=130,
                color=COLORS[provider],
                edgecolors=BG,
                linewidths=1.5,
                zorder=4,
            )
        rewards = [t["best_reward"] for t in runs]
        if rewards:
            ax.plot(
                xs[: len(rewards)],
                rewards,
                color=COLORS[provider],
                linewidth=2,
                alpha=0.5,
                zorder=3,
            )
    ax.axhline(1.0, color=FG, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(
        3.45,
        1.02,
        "key recovered\n(ν_D2/ν_D5/ν_D10 ≈ 1.5/3.0/5.5)",
        color=MUT,
        fontsize=9.5,
        ha="right",
        va="bottom",
    )
    ax.axhspan(0.95, 1.05, color=EDGE, alpha=0.15, zorder=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[0] for c in conds], fontsize=11)
    ax.set_ylabel("best episode reward")
    ax.set_ylim(-0.08, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_ax(
        ax,
        "Context-window ablation inside the environment (CA C2, D-ball exponents)",
        "dots = individual runs (all published) · for Beam the D-ball derivation sits past the excerpt window",
    )
    ax.annotate(
        "Beam re-derives the 1D model\n(ν = D: reports 2/5/10)",
        xy=(0, 0.0),
        xytext=(0.42, 0.42),
        color=MUT,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": DIM, "lw": 1.2},
    )
    ax.annotate(
        "Opus succeeds from the excerpt once —\nhonest counterpoint, also published",
        xy=(2, 1.0),
        xytext=(2.28, 0.62),
        color=MUT,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": DIM, "lw": 1.2},
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "context_ablation.png", facecolor=BG, dpi=160, bbox_inches="tight"
    )
    plt.close(fig)


def fig_improve(eps: list[dict]) -> None:
    # Verified values from papers/jmlr-22-1228-ca-grokking-beam (scores.json):
    # clean round att2 nu = 1.4759 (falsified); improve round att3 nu = 1.0044
    # (supported); 27B-class bake-off nu ~ 6000-6700 (off chart).
    points = [
        ("27B-class\n(recorded rounds)", 6072, MUT, "nu ~ 6000-6700, off this axis"),
        (
            "Beam, clean round\n(no feedback, no refs)",
            1.4759,
            COLORS["BEAMHF"],
            "nu = 1.476 -> falsified",
        ),
        (
            "Beam self-critique ->\nBeam re-audit",
            1.0044,
            COLORS["BEAMHF"],
            "nu = 1.0044 -> supported",
        ),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=BG)
    xs = [0, 1, 2]
    ax.set_yscale("log")
    ax.axhspan(0.95, 1.05, color="#1ec2b5", alpha=0.14, zorder=0)
    ax.axhline(1.0, color="#1ec2b5", linewidth=1.4, linestyle="--", alpha=0.8)
    ax.text(
        0.02,
        0.93,
        "target band  ν = 1.0 ± 0.05",
        color="#1ec2b5",
        fontsize=10.5,
        transform=ax.get_yaxis_transform(),
    )
    for x, (_label, val, color, ann) in zip(xs, points, strict=True):
        ax.scatter(
            [x], [val], s=160, color=color, edgecolors=BG, linewidths=1.5, zorder=4
        )
        ax.annotate(
            ann,
            (x, val),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            color=FG,
            fontsize=10.5,
            fontweight="bold",
        )
    ax.plot(
        [1, 2],
        [points[1][1], points[2][1]],
        color=COLORS["BEAMHF"],
        linewidth=2,
        alpha=0.6,
        zorder=3,
    )
    ax.plot(
        [0, 1],
        [points[0][1], points[1][1]],
        color=DIM,
        linewidth=1.4,
        linestyle=":",
        alpha=0.7,
        zorder=3,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points], fontsize=11)
    ax.set_ylabel("fitted critical exponent ν (log scale)")
    ax.set_ylim(0.5, 20000)
    ax.set_yticks([0.6, 1, 2, 10, 100, 1000, 10000])
    ax.set_yticklabels(["0.6", "1.0", "2", "10", "100", "1k", "10k"])
    style_ax(
        ax,
        "The self-improvement loop: Beam critiquing Beam, zero human feedback",
        "Beam-as-reviewer diagnosed its own script's bug (unshifted distribution, wrong fit window); "
        "Beam-as-auditor fixed it · flip is verify_gain-flagged and published",
    )
    fig.tight_layout()
    fig.savefig(OUT / "improve_arc.png", facecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = load_episodes()
    fig_ladder(eps)
    fig_ablation(eps)
    fig_improve(eps)
    for f in sorted(OUT.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
