#!/usr/bin/env python3
"""
Unsupervised threshold diagnostics (NO ground truth required).

On real schema-poor data there are no reference types, so the threshold must be
chosen from signals computable from the clustering alone. This sweeps the
threshold and reports, per threshold:
  - coverage, num_clusters, phases
  - avg internal similarity (cohesion) and avg inter-cluster similarity
  - separation score = internal - inter   (higher = tighter, better separated)
plus a dataset-level heterogeneity statistic: distinct phase-0 signatures /
entities (low = homogeneous, high = heterogeneous).

Usage:
    python -m type_abstraction.sweep_unsupervised data.nt --min 0.1 --max 0.9 --step 0.05
"""

import argparse
import os
import tempfile
from contextlib import redirect_stdout

import rdflib
from rdflib import RDF

import type_abstraction.type_abstractor as ta


def strip_types(path: str) -> str:
    g = rdflib.Graph()
    g.parse(path, format=rdflib.util.guess_format(path) or "nt")
    s = rdflib.Graph()
    for t in g:
        if t[1] != RDF.type:
            s.add(t)
    fd, tmp = tempfile.mkstemp(suffix=".nt")
    os.close(fd)
    s.serialize(destination=tmp, format="nt", encoding="utf-8")
    return tmp


def main():
    ap = argparse.ArgumentParser(description="Unsupervised threshold diagnostics.")
    ap.add_argument("input")
    ap.add_argument("--out", default=None, help="CSV path (default: <dataset>_unsup.csv)")
    ap.add_argument("--min", type=float, default=0.1)
    ap.add_argument("--max", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--pgf", action="store_true", help="Export .pgf-files for LaTeX (requires local TeX Live)")

    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_csv = args.out or f"{stem}_unsup.csv"
    out_png = f"{stem}_unsup.png"
    out_eps = f"{stem}_unsup.eps"
    out_pgf = f"{stem}_unsup.pgf"

    stripped = strip_types(args.input)
    ta._log = ta._Tee(os.devnull)
    engine = ta.IterativeTypeAbstraction(
        ta.IterConfig(max_iterations=args.max_iter, output_prefix=tempfile.mkdtemp()))
    with redirect_stdout(open(os.devnull, "w")):
        engine.load(stripped)

    # dataset heterogeneity: distinct phase-0 signatures / entities
    sig0 = engine._build_signatures(0)
    n = len(sig0)
    distinct = len({frozenset(s) for s in sig0.values()})
    diversity = distinct / n if n else 0.0
    print(f"entities (NR): {n} | distinct phase-0 signatures: {distinct} "
          f"| signature diversity: {diversity:.3f}")

    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    rows = []
    for th in thresholds:
        engine.cfg.similarity_threshold = th
        engine._entity2clusters.clear()
        engine._clusters = {}
        engine.history = []
        engine.vec = ta.FeatureVec()
        with redirect_stdout(open(os.devnull, "w")):
            engine.run()
        s = engine.history[-1]
        sil, ch, db = engine.final_internal_indices()
        rows.append((th, s.coverage, s.num_clusters, len(engine.history),
                     sil, ch, db))
        print(f"  t={th:.2f}  cov={s.coverage:.2f}  k={s.num_clusters}  "
              f"silhouette={sil:.3f}  CH={ch:.1f}  DB={db:.3f}")

    if ta._log is not None:
        ta._log.close()
        ta._log = None
    os.remove(stripped)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("threshold,coverage,num_clusters,phases,"
                "silhouette,calinski_harabasz,davies_bouldin\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Wrote {out_csv}")

    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from math import isfinite

        # EPS-sichere Font-Einbettung (Type 42 statt Type 3)
        plt.rcParams["ps.fonttype"] = 42
        plt.rcParams["pdf.fonttype"] = 42

        if args.pgf:
            plt.rcParams.update({
                        "pgf.texsystem": "pdflatex",
                        "text.usetex": True,
                        "pgf.rcfonts": False,  
                        "font.family": "serif",
                    })

        plt.rcParams.update({
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.markeredgewidth": 0,
        })

        th = [r[0] for r in rows]
        coverage    = [r[1] for r in rows]
        silhouette  = [r[4] for r in rows]
        ch_plot     = [r[5] if isfinite(r[5]) else float("nan") for r in rows]
        db_plot     = [r[6] if r[6] > 0 else float("nan") for r in rows]

        fig, ax1 = plt.subplots(figsize=(3.4, 2.7))
        cov_rescaled = [2 * c - 1 for c in coverage]

        l1, = ax1.plot(th, silhouette, "-o", color="#1f77b4", linewidth=1.1,
                        markersize=2.5, label="Silhouette (higher better)")
        l2, = ax1.plot(th, cov_rescaled, color="#444444", linewidth=1.0,
                        linestyle=(0, (4, 2)), label="Coverage (higher better)")
        ax1.set_xlabel(r"similarity threshold $\theta$")
        ax1.set_ylabel("Silhouette / Coverage*")
        ax1.set_ylim(-1, 1)
        ax1.axhline(0, color="0.85", linewidth=0.5, zorder=0)  # kein alpha!

        ax2 = ax1.twinx()
        l3, = ax2.plot(th, ch_plot, color="#2ca02c", linewidth=1.1,
                        linestyle=(0, (5, 1, 1, 1)), marker="^", markersize=2.5,
                        label="Calinski-Harabasz (higher better)")
        l4, = ax2.plot(th, db_plot, color="#d62728", linewidth=1.0,
                        linestyle=(0, (3, 2)), marker="s", markersize=2.5,
                        label="Davies-Bouldin (lower better)")
        ax2.set_yscale("log")
        ax2.set_ylabel("CH / DB (log)")

        # kein alpha beim Grid -> helle solide Farbe stattdessen
        ax1.grid(True, axis="both", color="0.88", linewidth=0.4)
        ax1.set_axisbelow(True)
        for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
            spine.set_color("0.4")

        lines = [l1, l2, l3, l4]
        # kein framealpha -> solide facecolor
        leg = ax1.legend(lines, [l.get_label() for l in lines], loc="lower left",
                        fontsize=6.5, edgecolor="0.7", facecolor="white")
        leg.get_frame().set_linewidth(0.5)

        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_eps, format="eps", bbox_inches="tight")

        if args.pgf:
            fig.savefig(out_pgf, format="pgf", bbox_inches="tight")
            print(f"Wrote {out_png}, {out_eps} and {out_pgf}")
        print(f"Wrote {out_png} and {out_eps}")


if __name__ == "__main__":
    main()