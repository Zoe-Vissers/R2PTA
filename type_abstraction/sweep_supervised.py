#!/usr/bin/env python3
"""
Threshold sweep for type-abstraction reconstruction validity.

Ground truth comes from one of two sources (--truth-mode):
  type : entity -> set of rdf:type class IRIs   (default)
  iri  : entity -> IRI local name with trailing digits removed
         (e.g. .../wsdbm/User0 -> 'User', .../Product123 -> 'Product')

For the chosen ground truth this script:
  1. reads the ground-truth types (entity -> set of class IRIs);
  2. removes all rdf:type triples to get a purely structural input;
  3. runs the abstraction across a grid of thresholds;
  4. scores each inferred partition against the truth partition with
     ARI / AMI / V-measure, and records coverage and cluster count;
  5. writes a CSV and (optionally) a PNG curve for the paper.

The evaluation set is the NR entities the abstractor actually considers (i.e.
that still have a structural signature after type removal) that also carry a
truth type. Entities whose only triples were type assertions have no structural
signature and are reported as out of scope, not as errors.

Ground-truth label per entity = the (frozen) set of its class IRIs; two entities are in
the same truth class iff they have identical type sets. Uncovered entities are
each given their own singleton predicted label, so low coverage is penalised.

Usage:
    python -m type_abstraction.sweep_supervised dataset.ttl --out ...
"""

import argparse
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout

import rdflib
from rdflib import RDF
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)

import type_abstraction.type_abstractor as ta


def derive_iri_label(iri: str) -> str:
    """Ground-truth type = final path/fragment segment with trailing digits
    removed. e.g. .../wsdbm/User0 -> 'User', .../Product123 -> 'Product'."""
    local = re.split(r"[/#]", iri)[-1]
    return re.sub(r"\d+$", "", local)


def read_ground_truth(data_graph, mode: str, truth_path=None):
    """Ground-truth entity->type(s). In 'type' mode the types are read from
    truth_path if given (a separate file, e.g. DBpedia instance-types),
    otherwise from data_graph itself. In 'iri' mode they are derived from the
    data_graph's IRIs."""
    if mode == "type":
        if truth_path is None:
            gt = data_graph
        else:
            gt = rdflib.Graph()
            gt.parse(truth_path, format=rdflib.util.guess_format(truth_path) or "nt")
        truth = {}
        for s, _, o in gt.triples((None, RDF.type, None)):
            truth.setdefault(str(s), set()).add(str(o))
        return {e: frozenset(cs) for e, cs in truth.items()}
    elif mode == "iri":
        truth = {}
        for node in set(data_graph.subjects()) | set(data_graph.objects()):
            if isinstance(node, rdflib.URIRef):
                lbl = derive_iri_label(str(node))
                if lbl:                      # skip empty labels (e.g. all-digit locals)
                    truth[str(node)] = lbl
        return truth
    else:
        raise ValueError(f"unknown truth mode {mode!r}")


def strip_types(g: rdflib.Graph) -> str:
    stripped = rdflib.Graph()
    for s, p, o in g:
        if p != RDF.type:
            stripped.add((s, p, o))
    fd, tmp = tempfile.mkstemp(suffix=".nt")
    os.close(fd)
    stripped.serialize(destination=tmp, format="nt", encoding="utf-8")
    return tmp


def build_engine(stripped_path: str, max_iter: int):
    cfg = ta.IterConfig(max_iterations=max_iter, output_prefix=tempfile.mkdtemp())
    engine = ta.IterativeTypeAbstraction(cfg)
    ta._log = ta._Tee(os.devnull)        # silence the engine
    with redirect_stdout(open(os.devnull, "w")):
        engine.load(stripped_path)       # parse + classify + profiles, once
    return engine


def run_at_threshold(engine, threshold: float):
    engine.cfg.similarity_threshold = threshold
    engine._entity2clusters.clear()      # reset per-run state; profiles are reused
    engine._clusters = {}
    engine.history = []
    engine.vec = ta.FeatureVec()
    with redirect_stdout(open(os.devnull, "w")):
        engine.run()
    snap = engine.history[-1]
    phases = len(engine.history)          # baseline + refinements actually run
    entity2cid = {e: cid for cid, members in snap.cluster_members.items()
                  for e in members}
    return set(engine.profiles), entity2cid, snap.num_clusters, phases


def score(eval_entities, truth, entity2cid):
    # truth label ids
    truth_key2id, truth_labels = {}, []
    pred_labels = []
    next_singleton = -1
    for e in sorted(eval_entities):
        gkey = truth[e]
        truth_labels.append(truth_key2id.setdefault(gkey, len(truth_key2id)))
        if e in entity2cid:
            pred_labels.append(entity2cid[e])
        else:                       # uncovered -> unique singleton cluster
            pred_labels.append(next_singleton)
            next_singleton -= 1
    h, c, v = homogeneity_completeness_v_measure(truth_labels, pred_labels)
    return (adjusted_rand_score(truth_labels, pred_labels),
            adjusted_mutual_info_score(truth_labels, pred_labels),
            h, c, v)


def main():
    ap = argparse.ArgumentParser(description="Threshold sweep vs ground-truth rdf:type.")
    ap.add_argument("input", help="Dataset to evaluate against ground truth")
    ap.add_argument("--truth-mode", choices=["type", "iri"], default="type",
                    help="'type': ground truth from rdf:type triples; "
                         "'iri': truth from IRI local name minus trailing digits")
    ap.add_argument("--truth-file", default=None,
                    help="Separate file holding the ground-truth rdf:type triples "
                         "(type mode only; e.g. DBpedia instance-types). "
                         "If omitted, types are read from the input file.")
    ap.add_argument("--out", default=None,
                    help="CSV path (default: <dataset>.csv)")
    ap.add_argument("--min", type=float, default=0.1)
    ap.add_argument("--max", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--pgf", action="store_true", help="Export .pgf-files for LaTeX (requires local TeX Live)")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_csv = args.out or f"{stem}.csv"
    out_png = f"{stem}.png"
    out_eps = f"{stem}.eps"
    out_pgf = f"{stem}.pgf"

    g = rdflib.Graph()
    g.parse(args.input, format=rdflib.util.guess_format(args.input) or "nt")
    truth = read_ground_truth(g, args.truth_mode, args.truth_file)
    stripped = strip_types(g)   # keep the method purely structural (no-op if input has no types)
    src = args.truth_file or args.input
    print(f"Ground-truth entities ({args.truth_mode}, from {os.path.basename(src)}): {len(truth)}")

    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    engine = build_engine(stripped, args.max_iter)   # parse + classify once
    rows = []
    for th in thresholds:
        nr, entity2cid, k, phases = run_at_threshold(engine, th)
        eval_entities = nr & set(truth)            # NR and ground-truth-labelled
        if not eval_entities:
            print(f"  t={th}: no evaluable entities", file=sys.stderr)
            continue
        ari, ami, hom, comp, vm = score(eval_entities, truth, entity2cid)
        covered = sum(1 for e in eval_entities if e in entity2cid)
        coverage = covered / len(eval_entities)
        rows.append((th, ari, ami, hom, comp, vm, coverage, k, phases,
                     len(eval_entities)))
        print(f"  t={th:.2f}  ARI={ari:.3f}  AMI={ami:.3f}  "
              f"hom={hom:.3f}  comp={comp:.3f}  V={vm:.3f}  "
              f"cov={coverage:.2f}  k={k}  phases={phases}")

    if ta._log is not None:
        ta._log.close()
        ta._log = None
    os.remove(stripped)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)      
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("threshold,ARI,AMI,homogeneity,completeness,Vmeasure,"
                "coverage,num_clusters,phases,eval_entities\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Wrote {out_csv}  ({len(rows)} rows)")
    if rows:
        evaluated = rows[0][9]
        print(f"Evaluated {evaluated} of {len(truth)} ground-truth entities "
              f"(rest are non-NR or have no structural signature)")

    if out_png and rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot", file=sys.stderr)
            return

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
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.markeredgewidth": 0,
        })

        th = [r[0] for r in rows]
        ari          = [r[1] for r in rows]
        ami          = [r[2] for r in rows]
        homogeneity  = [r[3] for r in rows]
        completeness = [r[4] for r in rows]
        vmeasure     = [r[5] for r in rows]
        coverage     = [r[6] for r in rows]

        fig, ax = plt.subplots(figsize=(3.4, 2.9))

        ax.plot(th, ari,          color="#1f77b4", linewidth=1.1, marker="o", markersize=2.5, label="ARI")
        ax.plot(th, ami,          color="#ff7f0e", linewidth=1.1, marker="s", markersize=2.5, label="AMI")
        ax.plot(th, vmeasure,     color="#2ca02c", linewidth=1.1, marker="^", markersize=2.5, label="V-measure")
        ax.plot(th, homogeneity,  color="#9467bd", linewidth=0.9, linestyle=(0, (3, 1.5)), label="Homogeneity")
        ax.plot(th, completeness, color="#8c564b", linewidth=0.9, linestyle=(0, (1, 1.3)), label="Completeness")
        ax.plot(th, coverage,     color="#444444", linewidth=1.1, linestyle=(0, (4, 2)), label="Coverage")

        ax.set_xlabel(r"similarity threshold $\theta$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)

        ax.grid(True, axis="both", color="0.88", linewidth=0.4)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("0.4")

        leg = ax.legend(loc="lower left", fontsize=8, edgecolor="0.7", facecolor="white",
                        framealpha=1.0, ncol=2)
        leg.get_frame().set_linewidth(0.5)

        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_eps, format="eps", bbox_inches="tight")
        if args.pgf:
            fig.savefig(out_pgf, format="pgf", bbox_inches="tight")
            print(f"Wrote {out_png}, {out_eps} and {out_pgf}")
        else: 
            print(f"Wrote {out_png} and {out_eps}")


if __name__ == "__main__":
    main()