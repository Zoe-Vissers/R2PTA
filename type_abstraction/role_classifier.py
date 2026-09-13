"""
RDF Role Classifier
===================
Classifies elements in an RDF graph into element-level roles (NR, NPVR)
and reports only the NR ("node-like") entities in the output.
Duplicate triples are detected and skipped before classification.
"""

from collections import defaultdict
import warnings
import rdflib
import rdflib.term as _rt
from rdflib import URIRef, BNode, Literal

_OrigLiteral = _rt.Literal
class _NoNormLiteral(_OrigLiteral):
    """Preserve the original lexical form of typed literals by disabling RDFLib's
       automatic normalization. The framework performs lookups by exact string
       equality rather than XSD value-space equivalence, so normalizing literals
       (e.g. "01"^^xsd:integer -> "1"^^xsd:integer or "1"^^xsd:boolean -> "true")
       would break these lookups and could lead to failed matches.
    """
    def __new__(cls, value, lang=None, datatype=None, normalize=None):
        return _OrigLiteral.__new__(cls, value, lang=lang, datatype=datatype,
                                    normalize=False)
_rt.Literal = _NoNormLiteral


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def classify(graph_input: str, k: int = 1) -> dict:
    """
    Classify element-level roles in an RDF graph (Rules 1-2 only) and
    return the NR ("node-like") entities.

    Parameters
    ----------
    graph_input : str
        RDF content (Turtle or N-Triples) or a file path.
    k : int
        Predicate-count threshold for NR-type-2 vs NPVR (default 1).

    Returns
    -------
    dict with keys: nr_entities, u_partitions, summary, configuration.
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    # -- Step 1: Parse --------------------------------------------------------
    g = _parse(graph_input)

    # -- Step 2: Deduplicate triples ------------------------------------------
    # rdflib graphs are already sets, but raw serialisations may contain dupes
    seen = set()
    triples = []
    duplicates_skipped = 0
    for t in g:
        if t in seen:
            duplicates_skipped += 1
        else:
            seen.add(t)
            triples.append(t)

    # -- Step 3: Build sets and helper quantities in one pass -----------------
    U_s, U_p, U_o = set(), set(), set()
    out_deg = defaultdict(int)   # out_deg[u]  = #triples where u is subject
    in_pred = defaultdict(set)   # in_pred[u]  = distinct predicates referencing u as object

    for s, p, o in triples:
        U_s.add(s); U_p.add(p); U_o.add(o)
        out_deg[s] += 1
        in_pred[o].add(p)

    U_so = U_s | U_o

    # -- Step 4: Disjointness check -------------------------------------------
    # Previously fatal (raised ValueError). Now only a non-fatal notice:
    # the overlap doesn't affect element-level classification below (that only
    # looks at U_so), so we flag it and continue instead of aborting.
    overlap = U_so & U_p
    if overlap:
        warnings.warn(
            f"Disjointness violated - IRIs in both subject/object and predicate position: {overlap}",
            stacklevel=2,
        )

    # -- Step 5: Element-level role assignment (Rules 1 & 2) ------------------
    NR_set, NPVR_set = set(), set()

    for u in U_so:
        od = out_deg[u]
        ip = len(in_pred[u])

        if not isinstance(u, Literal) and (od > 0 or ip > k):
            NR_set.add(u)          # Rule 1: node-like element
        else:
            NPVR_set.add(u)        # Rule 2: value-like element

    # -- Step 6: Element-level coverage check ---------------------------------
    # NPVR_set is only needed internally to prove every element in U_so was
    # classified exactly once; it is not surfaced in the output.
    _check_element_coverage(U_so, NR_set, NPVR_set)

    # -- Summary statistics (element-level) -----------------------------------
    summary = {
        "triples_total":      len(triples) + duplicates_skipped,
        "triples_unique":     len(triples),
        "duplicates_skipped": duplicates_skipped,
        "nodes_U_so":         len(U_so),
        "predicates_U_p":     len(U_p),
        "NR":                 len(NR_set),
        "NPVR":                len(NPVR_set),
    }

    result = {
        "nr_entities": [_key(u) for u in NR_set],
        "u_partitions": {
            "U_s":  [_key(u) for u in U_s],
            "U_p":  [_key(u) for u in U_p],
            "U_o":  [_key(u) for u in U_o],
            "U_so": [_key(u) for u in U_so],
        },
        "summary": summary,
        "configuration": {"k": k},
    }

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(graph_input: str) -> rdflib.Graph:
    """Parse Turtle/N-Triples from a file path or raw string."""
    g = rdflib.Graph()
    try:
        g.parse(graph_input)
    except Exception:
        fmt = "turtle" if ("@prefix" in graph_input or "@base" in graph_input) else "nt"
        g.parse(data=graph_input, format=fmt)
    return g


def _key(node) -> str:
    """Stable string key for an RDF node.

    IRIs and blank nodes return as-is. Literals are wrapped with outer
    quote characters so they are unambiguously distinguishable from
    IRIs in unit sets (an IRI classified as NPVR would otherwise be
    indistinguishable from a literal whose value happens to look like
    an IRI). 
    """
    if isinstance(node, URIRef): return str(node)
    if isinstance(node, BNode):  return f"_:{node}"
    lex = str(node)
    if len(lex) >= 2 and lex.startswith('"') and lex.endswith('"'):
        return lex  # already quote-wrapped (lexical value contains outer quotes)
    return f'"{lex}"'


def _check_element_coverage(U_so, NR_set, NPVR_set):
    """Raise if any element in U_so is unclassified or multiply classified."""
    classified = NR_set | NPVR_set
    if classified != U_so:
        raise RuntimeError(
            f"Element-level coverage failure. "
            f"Missing: {U_so - classified}. Extra: {classified - U_so}."
        )
    if NR_set & NPVR_set:
        raise RuntimeError("Element-level coverage failure: overlapping role sets.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys, os

    parser = argparse.ArgumentParser(
        description="Classify NR (node-like) entities in an RDF graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph",   help="Path to RDF file (.ttl, .nt, .rdf, ...)")
    parser.add_argument("--k",     type=int, default=1,        help="NR predicate threshold")
    parser.add_argument("--out",   default="-",  help="Main output file path (default: stdout)")
    parser.add_argument("--stats", default=None, help="Stats output file (auto-derived from --out if omitted)")
    args = parser.parse_args()

    result = classify(args.graph, k=args.k)

    # -- Determine stats output path ------------------------------------------
    # If --stats not given and --out is a real file, auto-derive: foo.json -> foo_stats.json
    stats_path = args.stats
    if stats_path is None and args.out != "-":
        base, ext = os.path.splitext(args.out)
        stats_path = f"{base}_stats{ext}"

    # -- Write summary stats to its own file ----------------------------------
    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(result["summary"], f, indent=2)
        print(f"Summary statistics -> {stats_path}", file=sys.stderr)

    # -- Write main output (summary omitted - it lives in the stats file) -----
    if args.out == "-":
        # When printing to stdout, include summary inline
        print(json.dumps(result, indent=2))
    else:
        output_data = {k: v for k, v in result.items() if k != "summary"}
        with open(args.out, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results           -> {args.out}", file=sys.stderr)