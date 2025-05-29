#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter, MaxNLocator

FIGSIZE_MM   = (160, 100)
DPI          = 300
MARKER       = "o"
LINESTYLE    = "-"
XTICK_ROT    = 30
XTICK_FSIZE  = 8

def _mm_to_in(mm: float) -> float:
    return mm / 25.4


def _first(df: pd.DataFrame, *cands: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"CSV missing column(s): {', '.join(cands)}")


def _plain_numbers(ax):
    for axis in (ax.xaxis, ax.yaxis):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _set_all_xticks(ax, vals):
    vals = sorted(vals)
    ax.set_xticks(vals)
    ax.set_xticklabels(
        [str(v) for v in vals],
        rotation=XTICK_ROT,
        ha="right",
        fontsize=XTICK_FSIZE,
    )


def _set_dense_yticks(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12, integer=True, prune=None))


def _plot_multi(ax, df, y, ylabel):
    for backend, grp in df.groupby("backend"):
        g = grp.sort_values("concurrency")
        ax.plot(g["concurrency"], g[y],
                marker=MARKER, linestyle=LINESTYLE, label=backend)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs concurrency")
    ax.grid(True, which="both", ls=":")
    ax.legend(title="Backend")

    _plain_numbers(ax)
    _set_all_xticks(ax, df["concurrency"].unique())
    _set_dense_yticks(ax)


def _plot_single(ax, df, y, ylabel, colour="tab:blue"):
    g = df.sort_values("concurrency")
    ax.plot(g["concurrency"], g[y],
            marker=MARKER, linestyle=LINESTYLE, color=colour)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs concurrency")
    ax.grid(True, which="both", ls=":")

    _plain_numbers(ax)
    _set_all_xticks(ax, df["concurrency"].unique())
    _set_dense_yticks(ax)


def main(csv: Path, fmt="png"):
    df = pd.read_csv(csv)

    rps  = _first(df, "requests_s", "rps")
    tps  = _first(df, "tokens_s",   "tps")
    avg  = _first(df, "latency_avg_s", "avg_latency_s")
    p95  = _first(df, "latency_p95_s", "p95_latency_s")

    base = df.loc[df["concurrency"] == 1, ["backend", rps]].set_index("backend")[rps]
    df["speedup"]      = df.apply(lambda r: r[rps] / base[r["backend"]], axis=1)
    df["parallel_eff"] = df["speedup"] / df["concurrency"]
    df["tail_ratio"]   = df[p95] / df[avg]

    # relative throughput vLLM / Ollama
    pivot = df.pivot(index="concurrency", columns="backend", values=rps)
    rel = None
    if {"vLLM", "Ollama"}.issubset(pivot.columns):
        pivot = pivot.dropna(subset=["vLLM", "Ollama"])
        if not pivot.empty:
            rel = (
                pivot.assign(rel=pivot["vLLM"] / pivot["Ollama"])
                     .reset_index()[["concurrency", "rel"]]
            )

    figsize = (_mm_to_in(FIGSIZE_MM[0]), _mm_to_in(FIGSIZE_MM[1]))
    series = [
        (rps,  "Completed Requests/s",               f"rps_vs_concurrency.{fmt}"),
        (tps,  "Tokens/s",                 f"tps_vs_concurrency.{fmt}"),
        (avg,  "Average latency (s)",        f"latency_avg_vs_concurrency.{fmt}"),
        (p95,  "p95 latency (s)",            f"latency_p95_vs_concurrency.{fmt}"),
        ("speedup",        "Speed-up (× vs 1-way)",
                                             f"speedup_vs_concurrency.{fmt}"),
        ("parallel_eff",   "Parallel efficiency",
                                             f"efficiency_vs_concurrency.{fmt}"),
        ("tail_ratio",     "p95 / mean latency",
                                             f"tail_ratio_vs_concurrency.{fmt}"),
    ]

    for col, label, fname in series:
        fig, ax = plt.subplots(figsize=figsize)
        _plot_multi(ax, df, col, label)
        fig.tight_layout()
        fig.savefig(fname, dpi=DPI if fmt == "png" else None)
        plt.close(fig)
        print(f"✓ wrote {fname}")

    if rel is not None:
        fig, ax = plt.subplots(figsize=figsize)
        _plot_single(ax, rel, "rel", "Throughput ratio  (vLLM ÷ Ollama)")
        fig.tight_layout()
        fname = f"relative_throughput_vllm_over_ollama.{fmt}"
        fig.savefig(fname, dpi=DPI if fmt == "png" else None)
        plt.close(fig)
        print(f"✓ wrote {fname}")
    else:
        print("⚠ Skipped relative-throughput plot – need both backends.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate benchmark plots.")
    ap.add_argument("csv", help="benchmark-results.csv")
    ap.add_argument("-f", "--format", default="png",
                    choices=("png", "pdf", "svg"), help="output format")
    args = ap.parse_args()
    main(Path(args.csv), fmt=args.format)
