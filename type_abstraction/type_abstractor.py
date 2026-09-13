#!/usr/bin/env python3
"""
Iterative RDF Type Abstraction

Groups RDF entities that play an *entity-like* role (role NR, as decided by the
role classifier) into clusters based on their structural signature, then emits
one rdf:type triple per clustered entity. Intended for schema-poor datasets.

Signature of an entity (a set of features):

  Phase 0 (Baseline)
    OUT:<pred>, IN:<pred>            -- outgoing / incoming predicate presence.

  Phase N>0 (Refinement; Weisfeiler-Leman-style)
    For each neighbour reached via predicate p that is itself an NR entity with
    inferred cluster CID, add:
        OUT:p|NR_TYPE:CID   /   IN:p|NR_TYPE:CID
    (Literals and non-NR resources contribute nothing: type is about the
    structure of an entity and the types of its neighbours, not their identity.)

Similarity is IDF-weighted cosine over these features. Predicates shared by
*every* entity carry no information (IDF = 0) and are dropped automatically.

Clustering: an edge is drawn between two entities when their similarity >= 
threshold, and each connected component (of size >= min_cluster_size, whose 
members share >= 1 informative feature) becomes a type.

Convergence: stop once the partition is stable (fraction of entities whose
co-membership set changes <= epsilon), measured invariantly to relabelling.

Input formats: anything rdflib can parse (.nt, .ttl, .n3, .nq ...); the format
is guessed from the file extension.

Usage:
    python -m type_abstraction.type_abstractor \
    --threshold 0.6 --output run1 /path/to/dataset.ttl
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, islice
from math import log as math_log, sqrt
from typing import Dict, List, Optional, Set, Tuple

import rdflib
from rdflib import RDF

from type_abstraction.role_classifier import classify

# ─────────────────────────────────────────────────────────────────────────────
# Logging: verbose detail -> .log file; stdout gets one-line status only.
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, log_path: str):
        self._f = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg: str):
        self._f.write(msg)

    def close(self):
        self._f.close()


_log: Optional[_Tee] = None


def log(msg: str = "", end: str = "\n"):
    line = msg + end
    if _log is not None:
        _log.write(line)
    else:
        sys.stdout.write(line)


def progress(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    if _log is not None:
        _log.write(msg + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (threshold is the only thing to tune per dataset)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IterConfig:
    similarity_threshold: float = 0.5
    min_cluster_size: int = 2
    max_iterations: int = 5
    convergence_epsilon: float = 0.01
    output_prefix: str = "iterative"
    save_all_iterations: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Entity profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityProfile:
    entity: str
    outgoing: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    incoming: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))


# ─────────────────────────────────────────────────────────────────────────────
# Per-iteration snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterSnapshot:
    iteration: int
    num_clusters: int
    coverage: float
    avg_internal_sim: float
    avg_cluster_size: float
    min_cluster_size: int
    max_cluster_size: int
    reassignment_rate: float
    cluster_members: Dict[int, List[str]]
    cluster_shared_out: Dict[int, List[str]]
    cluster_shared_in: Dict[int, List[str]]
    cluster_shared_features: Dict[int, List[str]] = field(default_factory=dict)
    avg_inter_cluster_sim: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# IDF-weighted feature vectors + cosine
# ─────────────────────────────────────────────────────────────────────────────

class FeatureVec:
    """Maps feature strings to ids, holds per-feature IDF weights, and computes
    cosine similarity over IDF-weighted vectors.

    IDF(f) = log(N / df_f). A feature present in every entity has IDF = 0 and is
    dropped (it cannot help distinguish anything)."""

    def __init__(self):
        self._f2id: Dict[str, int] = {}
        self._next = 0
        self._idf: Dict[str, float] = {}

    def _id(self, f: str) -> int:
        if f not in self._f2id:
            self._f2id[f] = self._next
            self._next += 1
        return self._f2id[f]

    def build_idf(self, all_feature_sets: List[Set[str]]):
        n = len(all_feature_sets)
        df: Dict[str, int] = defaultdict(int)
        for fs in all_feature_sets:
            for f in fs:
                df[f] += 1
        self._idf = {f: math_log(n / d) for f, d in df.items() if d > 0}

    def informative(self, features: Set[str]) -> Set[str]:
        """Drop features with IDF == 0 (present in every entity)."""
        return {f for f in features if self._idf.get(f, 0.0) > 0.0}

    def encode(self, features: Set[str]) -> Dict[int, float]:
        return {self._id(f): w for f in features
                if (w := self._idf.get(f, 0.0)) > 0.0}

    @staticmethod
    def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[k] * b[k] for k in (set(a) & set(b)))
        na = sqrt(sum(v * v for v in a.values()))
        nb = sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class IterativeTypeAbstraction:

    def __init__(self, cfg: IterConfig):
        self.cfg = cfg
        self.graph: Optional[rdflib.Graph] = None
        self.profiles: Dict[str, EntityProfile] = {}
        self.vec = FeatureVec()
        self._entity2clusters: Dict[str, Set[int]] = defaultdict(set)
        self._clusters: Dict[int, Set[str]] = {}
        self.history: List[ClusterSnapshot] = []
        self._nr_entities: Set[str] = set()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, path: str):
        """Parse the dataset with rdflib (format guessed from extension),
        classify roles, and build a profile for every NR entity."""
        log(f"\n[Load] Parsing {path} ...")
        t0 = time.time()

        fmt = rdflib.util.guess_format(path) or "nt"
        g = rdflib.Graph()
        g.parse(path, format=fmt)
        self.graph = g
        log(f"  {len(g):,} triples parsed as {fmt!r} in {time.time()-t0:.1f}s")

        log("  Calling role_classifier.classify() ...")
        t1 = time.time()
        classification = classify(path, k=1)
        nr_entities: List[str] = classification["nr_entities"]
        self._nr_entities = set(nr_entities)
        log(f"  classify() returned {classification['summary']['nodes_U_so']:,} elements "
            f"({len(self._nr_entities):,} NR) in {time.time()-t1:.1f}s")

        for s, p, o in g:
            s_str, p_str, o_str = str(s), str(p), str(o)
            if s_str in self._nr_entities:
                self.profiles.setdefault(s_str, EntityProfile(s_str)).outgoing[p_str].add(o_str)
            if o_str in self._nr_entities:
                self.profiles.setdefault(o_str, EntityProfile(o_str)).incoming[p_str].add(s_str)

        log(f"  {len(self.profiles)} entity profiles built | "
            f"total load time {time.time()-t0:.1f}s")

    # ── Feature building ──────────────────────────────────────────────────────

    def _base_features(self, entity: str) -> Set[str]:
        profile = self.profiles[entity]
        features = {f"OUT:{p}" for p in profile.outgoing}
        features |= {f"IN:{p}" for p in profile.incoming}
        return features

    def _enriched_features(self, entity: str) -> Set[str]:
        features = self._base_features(entity)
        profile = self.profiles[entity]
        for pred, objects in profile.outgoing.items():
            for obj in objects:
                for cid in self._entity2clusters.get(obj, ()):  # NR neighbours only
                    features.add(f"OUT:{pred}|NR_TYPE:{cid}")
        for pred, subjects in profile.incoming.items():
            for subj in subjects:
                for cid in self._entity2clusters.get(subj, ()):
                    features.add(f"IN:{pred}|NR_TYPE:{cid}")
        return features

    def _build_signatures(self, phase: int) -> Dict[str, Set[str]]:
        fn = self._base_features if phase == 0 else self._enriched_features
        return {e: fn(e) for e in self.profiles}

    # ── Clustering (IDF-cosine threshold graph) ─
    #
    # Entities with identical signatures are collapsed into one node before the
    # graph is built. This is lossless (identical signatures are always similarity 
    # 1 and always co-cluster) and is what keeps regular datasets like BSBM with 
    # thousands of structurally identical entities from producing an O(n^2) edge set.

    def _cluster(self, sigs: Dict[str, Set[str]]) -> Tuple[Dict[int, Set[str]],
                                                           Dict[str, Set[str]]]:
        self.vec.build_idf(list(sigs.values()))
        eff = {e: self.vec.informative(fs) for e, fs in sigs.items()}

        # Group entities by identical effective signature; drop empty signatures
        # (an entity with no informative feature has no basis for a type).
        groups: Dict[frozenset, List[str]] = defaultdict(list)
        for e, fs in eff.items():
            if fs:
                groups[frozenset(fs)].append(e)

        nodes = sorted(groups.items(), key=lambda kv: min(kv[1]))  # determinism
        sigsets = [set(sig) for sig, _ in nodes]
        members = [mem for _, mem in nodes]
        vecs = [self.vec.encode(s) for s in sigsets]
        u = len(nodes)

        idx: Dict[str, Set[int]] = defaultdict(set)
        for i, s in enumerate(sigsets):
            for f in s:
                idx[f].add(i)

        graph: Dict[int, Set[int]] = defaultdict(set)
        compared = 0
        for i in range(u):
            cand: Set[int] = set()
            for f in sigsets[i]:
                cand |= idx[f]
            for j in cand:
                if j <= i:
                    continue
                compared += 1
                if self.vec.cosine(vecs[i], vecs[j]) >= self.cfg.similarity_threshold:
                    graph[i].add(j)
                    graph[j].add(i)

        log(f"    Distinct signatures: {u:,} (from {len(eff):,} entities) | "
            f"Comparisons: {compared:,}")
        clusters = self._connected_components(u, graph, sigsets, members)
        return clusters, eff

    def _connected_components(
        self,
        u: int,
        graph: Dict[int, Set[int]],
        sigsets: List[Set[str]],
        members: List[List[str]],
    ) -> Dict[int, Set[str]]:
        cfg = self.cfg
        visited: Set[int] = set()
        clusters: Dict[int, Set[str]] = {}
        cid = 0
        for start in range(u):
            if start in visited:
                continue
            comp: List[int] = []
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                stack.extend(graph[cur] - visited)

            mem: Set[str] = set()
            for i in comp:
                mem.update(members[i])
            if len(mem) < cfg.min_cluster_size:
                continue
            shared = set(sigsets[comp[0]])
            for i in comp[1:]:
                shared &= sigsets[i]
            if not shared:
                log(f"    [SKIP] {len(mem)}-entity component: no shared informative feature")
                continue
            clusters[cid] = mem
            cid += 1
        return clusters

    # ── Convergence (relabel-invariant, linear) ───────────────────────────────

    @staticmethod
    def _reassignment_rate(
        prev_clusters: Dict[int, Set[str]],
        clusters: Dict[int, Set[str]],
        universe: Set[str],
    ) -> float:
        """Fraction of entities whose cluster (as a member-set) changed between
        phases, invariant to how clusters are numbered. O(n), not O(sum size^2):
        each distinct member-set gets a canonical id shared across both phases,
        and we compare one integer per entity."""
        if not prev_clusters:
            return 1.0
        canon: Dict[frozenset, int] = {}

        def label_map(clusters: Dict[int, Set[str]]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for members in clusters.values():
                cid = canon.setdefault(frozenset(members), len(canon))
                for e in members:
                    out[e] = cid
            return out

        prev_map = label_map(prev_clusters)
        new_map = label_map(clusters)
        n = len(universe)
        if not n:
            return 0.0
        changed = sum(1 for e in universe
                      if prev_map.get(e, -1) != new_map.get(e, -1))
        return changed / n

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_snapshot(
        self,
        iteration: int,
        clusters: Dict[int, Set[str]],
        prev_clusters: Dict[int, Set[str]],
        eff: Dict[str, Set[str]],
    ) -> ClusterSnapshot:
        n = len(self.profiles)
        n_typed = sum(1 for e in self.profiles if e in self._entity2clusters)

        reassign_rate = self._reassignment_rate(
            prev_clusters, clusters, set(self.profiles))

        vecs = {e: self.vec.encode(eff[e]) for e in eff}
        internal_sims: List[float] = []
        cluster_shared_out: Dict[int, List[str]] = {}
        cluster_shared_in: Dict[int, List[str]] = {}
        cluster_shared_features: Dict[int, List[str]] = {}

        for cid, members in clusters.items():
            mlist = sorted(members)

            sh_out = set(self.profiles[mlist[0]].outgoing)
            sh_in = set(self.profiles[mlist[0]].incoming)
            shared_feats = set(eff[mlist[0]])
            for m in mlist[1:]:
                sh_out &= set(self.profiles[m].outgoing)
                sh_in &= set(self.profiles[m].incoming)
                shared_feats &= eff[m]
            cluster_shared_out[cid] = sorted(sh_out)
            cluster_shared_in[cid] = sorted(sh_in)
            cluster_shared_features[cid] = sorted(shared_feats)

            # Deterministic similarity sample: first 50 pairs, generated lazily
            # (never materialise all O(size^2) pairs of a large cluster).
            for e1, e2 in islice(combinations(mlist, 2), 50):
                if vecs.get(e1) and vecs.get(e2):
                    internal_sims.append(self.vec.cosine(vecs[e1], vecs[e2]))

        avg_inter = self._inter_cluster_sim(clusters, vecs)
        sizes = [len(m) for m in clusters.values()]

        return ClusterSnapshot(
            iteration=iteration,
            num_clusters=len(clusters),
            coverage=n_typed / n if n else 0.0,
            avg_internal_sim=(sum(internal_sims) / len(internal_sims)
                              if internal_sims else 0.0),
            avg_cluster_size=sum(sizes) / len(sizes) if sizes else 0.0,
            min_cluster_size=min(sizes) if sizes else 0,
            max_cluster_size=max(sizes) if sizes else 0,
            reassignment_rate=reassign_rate,
            cluster_members={cid: sorted(m) for cid, m in clusters.items()},
            cluster_shared_out=cluster_shared_out,
            cluster_shared_in=cluster_shared_in,
            cluster_shared_features=cluster_shared_features,
            avg_inter_cluster_sim=avg_inter,
        )

    def _inter_cluster_sim(
        self,
        clusters: Dict[int, Set[str]],
        vecs: Dict[str, Dict[int, float]],
    ) -> float:
        if len(clusters) < 2:
            return 0.0

        centroids: Dict[int, Dict[int, float]] = {}
        for cid, members in clusters.items():
            acc: Dict[int, float] = defaultdict(float)
            for m in members:
                for k, v in vecs.get(m, {}).items():
                    acc[k] += v
            sz = len(members)
            centroids[cid] = {k: v / sz for k, v in acc.items()}

        cids = sorted(clusters)
        pairs = list(islice(combinations(cids, 2), 100))  # deterministic cap
        sims = [self.vec.cosine(centroids[a], centroids[b]) for a, b in pairs]
        return sum(sims) / len(sims) if sims else 0.0

    def final_internal_indices(self):
        """Silhouette / Calinski-Harabasz / Davies-Bouldin on the FINAL
        clustering, computed on demand (no per-phase cost, so it never runs
        during normal abstraction). All three are delegated to scikit-learn's
        reference implementations."""
        if not self.history:
            return float("nan"), float("nan"), float("nan")
        final_phase = self.history[-1].iteration
        sigs = self._build_signatures(final_phase)
        eff = {e: self.vec.informative(fs) for e, fs in sigs.items()}
        vecs = {e: self.vec.encode(eff[e]) for e in eff}
        res = self._sklearn_indices(self._clusters, vecs)
        return res if res is not None else (float("nan"),) * 3

    @staticmethod
    def _sklearn_indices(clusters, vecs):
        """Silhouette, Calinski-Harabasz and Davies-Bouldin via scikit-learn, on
        the same L2-normalised IDF-weighted vectors (Euclidean on unit vectors is
        a monotone function of cosine, matching the clustering geometry). Returns
        None if scikit-learn is unavailable (so the caller can fall back)."""
        try:
            import numpy as np
            from sklearn.metrics import (silhouette_score,
                                         calinski_harabasz_score,
                                         davies_bouldin_score)
        except ImportError:
            return None
        ent2cid = {e: cid for cid, ms in clusters.items() for e in ms}
        ents = [e for e in ent2cid if vecs.get(e)]
        cids = {ent2cid[e] for e in ents}
        NAN = float("nan")
        if len(cids) < 2 or len(ents) <= len(cids):
            return NAN, NAN, NAN
        feats = sorted({f for e in ents for f in vecs[e]})
        fidx = {f: i for i, f in enumerate(feats)}
        X = np.zeros((len(ents), len(feats)))
        for r, e in enumerate(ents):
            for f, w in vecs[e].items():
                X[r, fidx[f]] = w
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X /= norms
        labels = np.array([ent2cid[e] for e in ents])
        return (float(silhouette_score(X, labels, metric="euclidean")),
                float(calinski_harabasz_score(X, labels)),
                float(davies_bouldin_score(X, labels)))


    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cfg = self.cfg
        log("\n" + "=" * 64)
        log("ITERATIVE TYPE ABSTRACTION")
        log("=" * 64)

        prev_clusters: Dict[int, Set[str]] = {}

        for phase in range(cfg.max_iterations + 1):
            label = "Baseline" if phase == 0 else f"Refinement {phase}"
            progress(f"  -> Phase {phase} ({label}) ...")
            log(f"\n{'─'*64}")
            log(f"  Phase {phase}  ({label})")
            log(f"{'─'*64}")
            t0 = time.time()

            log("  Building feature vectors ...")
            sigs = self._build_signatures(phase)

            log("  Clustering ...")
            clusters, eff = self._cluster(sigs)

            self._clusters = clusters
            self._entity2clusters.clear()
            for cid, members in clusters.items():
                for m in members:
                    self._entity2clusters[m].add(cid)

            snap = self._compute_snapshot(phase, clusters, prev_clusters, eff)
            self.history.append(snap)
            self._log_snapshot(snap, time.time() - t0)

            if cfg.save_all_iterations:
                self._save_iteration(phase, snap)

            if phase > 0 and snap.reassignment_rate <= cfg.convergence_epsilon:
                msg = (f"  Converged  (reassignment {snap.reassignment_rate:.3%}"
                       f" <= e={cfg.convergence_epsilon:.3%})")
                log(f"\n{msg}")
                progress(msg)
                break

            prev_clusters = clusters

    # ── Output helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _log_snapshot(snap: ClusterSnapshot, elapsed: float):
        log(f"\n  {'Metric':<38} {'Value':>10}")
        log(f"  {'─'*50}")
        sep = snap.avg_internal_sim - snap.avg_inter_cluster_sim
        for label, val in [
            ("Clusters", f"{snap.num_clusters}"),
            ("Coverage", f"{snap.coverage:.1%}"),
            ("Avg cluster size", f"{snap.avg_cluster_size:.1f}"),
            ("Min / Max cluster size", f"{snap.min_cluster_size} / {snap.max_cluster_size}"),
            ("Avg internal similarity", f"{snap.avg_internal_sim:.4f}"),
            ("Avg inter-cluster sim", f"{snap.avg_inter_cluster_sim:.4f}"),
            ("Separation score", f"{sep:.4f}"),
            ("Reassignment rate", f"{snap.reassignment_rate:.2%}"),
            ("Wall time", f"{elapsed:.1f}s"),
        ]:
            log(f"  {label:<38} {val:>10}")

    def _save_iteration(self, phase: int, snap: ClusterSnapshot):
        self._write_nt(f"{self.cfg.output_prefix}_phase{phase}.nt", snap)
        self._write_report(f"{self.cfg.output_prefix}_phase{phase}_report.json", snap)
        log(f"  Saved phase {phase} files.")

    def save_final(self):
        if not self.history:
            log("No iterations completed - nothing to save.")
            return
        snap = self.history[-1]
        self._write_nt(f"{self.cfg.output_prefix}_final.nt", snap)
        self._write_report(f"{self.cfg.output_prefix}_final_report.json", snap)
        self._write_history(f"{self.cfg.output_prefix}_history.json")

        outdir = self.cfg.output_prefix
        log(f"\nOutputs written to ./{outdir}/")
        for name in ("_final.nt", "_final_report.json", "_history.json", "_run.log"):
            log(f"  {outdir}{name}")

    def _log_history_table(self):
        if not self.history:
            return
        log("\n" + "=" * 78)
        log("ITERATION HISTORY")
        log("=" * 78)
        log(f"{'Phase':<11} {'Clusters':>9} {'Coverage':>10} "
            f"{'AvgSize':>9} {'AvgSim':>9} {'Reassigned':>11}")
        log("─" * 78)
        for snap in self.history:
            label = "Baseline" if snap.iteration == 0 else f"Refine {snap.iteration}"
            log(f"{label:<11} {snap.num_clusters:>9} "
                f"{snap.coverage:>9.1%} "
                f"{snap.avg_cluster_size:>9.1f} "
                f"{snap.avg_internal_sim:>9.4f} "
                f"{snap.reassignment_rate:>10.2%}")

    def _outdir(self) -> str:
        os.makedirs(self.cfg.output_prefix, exist_ok=True)
        return self.cfg.output_prefix

    def _write_nt(self, path: str, snap: ClusterSnapshot):
        """Serialize the original graph (format-correct) and append the inferred
        rdf:type triples."""
        dest = os.path.join(self._outdir(), os.path.basename(path))
        self.graph.serialize(destination=dest, format="nt", encoding="utf-8")

        type_pred = str(RDF.type)

        def term(t: str) -> str:
            return t if t.startswith("_:") else f"<{t}>"

        with open(dest, "a", encoding="utf-8") as f:
            for cid, members in snap.cluster_members.items():
                tcls = f"http://inferred.type/Type{cid}"
                for e in members:
                    f.write(f"{term(e)} <{type_pred}> <{tcls}> .\n")

    def _write_report(self, path: str, snap: ClusterSnapshot):
        all_entities = set(self.profiles)
        assigned = {e for members in snap.cluster_members.values() for e in members}
        uncovered = sorted(all_entities - assigned)

        dest = os.path.join(self._outdir(), os.path.basename(path))
        with open(dest, "w", encoding="utf-8") as f:
            json.dump({
                "iteration": snap.iteration,
                "config": {
                    "similarity_threshold": self.cfg.similarity_threshold,
                    "similarity_metric": "idf_cosine",
                    "min_cluster_size": self.cfg.min_cluster_size,
                    "max_iterations": self.cfg.max_iterations,
                    "convergence_epsilon": self.cfg.convergence_epsilon,
                },
                "metrics": {
                    "num_clusters": snap.num_clusters,
                    "coverage": snap.coverage,
                    "avg_internal_sim": snap.avg_internal_sim,
                    "avg_inter_cluster_sim": snap.avg_inter_cluster_sim,
                    "avg_cluster_size": snap.avg_cluster_size,
                    "min_cluster_size": snap.min_cluster_size,
                    "max_cluster_size": snap.max_cluster_size,
                    "reassignment_rate": snap.reassignment_rate,
                    "num_uncovered": len(uncovered),
                },
                "uncovered_entities": uncovered,
                "clusters": [
                    {
                        "cluster_id": cid,
                        "type_iri": f"<http://inferred.type/Type{cid}>",
                        "size": len(members),
                        "members": members,
                        "shared_outgoing_preds": snap.cluster_shared_out.get(cid, []),
                        "shared_incoming_preds": snap.cluster_shared_in.get(cid, []),
                        "shared_features": snap.cluster_shared_features.get(cid, []),
                    }
                    for cid, members in snap.cluster_members.items()
                ],
            }, f, indent=2)

    def _write_history(self, path: str):
        dest = os.path.join(self._outdir(), os.path.basename(path))
        with open(dest, "w", encoding="utf-8") as f:
            json.dump([{
                "iteration": s.iteration,
                "num_clusters": s.num_clusters,
                "coverage": s.coverage,
                "avg_internal_sim": s.avg_internal_sim,
                "avg_inter_cluster_sim": s.avg_inter_cluster_sim,
                "avg_cluster_size": s.avg_cluster_size,
                "reassignment_rate": s.reassignment_rate,
            } for s in self.history], f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Iterative RDF type abstraction (IDF-cosine, single-linkage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="Input RDF file (.nt, .ttl, .n3, .nq ...)")
    ap.add_argument("--threshold", "-t", type=float, default=0.5,
                    help="IDF-cosine similarity threshold for drawing an edge")
    ap.add_argument("--min-size", type=int, default=2,
                    help="Minimum cluster size (singletons are never typed)")
    ap.add_argument("--max-iter", "-n", type=int, default=5)
    ap.add_argument("--epsilon", "-e", type=float, default=0.01)
    ap.add_argument("--output", "-o", default=None,
                    help="Output directory (created if missing; default: <dataset>_t<threshold>)")
    ap.add_argument("--save-all", action="store_true",
                    help="Write NT + report for every phase, not just the final one")
    args = ap.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{stem}_t{args.threshold}"

    cfg = IterConfig(
        similarity_threshold=args.threshold,
        min_cluster_size=args.min_size,
        max_iterations=args.max_iter,
        convergence_epsilon=args.epsilon,
        output_prefix=args.output,
        save_all_iterations=args.save_all,
    )

    os.makedirs(cfg.output_prefix, exist_ok=True)
    stem = os.path.basename(os.path.normpath(cfg.output_prefix))       
    log_path = os.path.join(cfg.output_prefix, f"{stem}_run.log")
    global _log
    _log = _Tee(log_path)

    progress(f"Logging to {log_path}")

    log("=" * 64)
    log("CONFIGURATION")
    log("=" * 64)
    log(f"  Input:               {args.input}")
    log(f"  Similarity:          IDF-weighted cosine")
    log(f"  Threshold:           {cfg.similarity_threshold}")
    log(f"  Min cluster size:    {cfg.min_cluster_size}")
    log(f"  Max iterations:      {cfg.max_iterations}")
    log(f"  Convergence e:       {cfg.convergence_epsilon:.3%}")
    log(f"  Save all phases:     {cfg.save_all_iterations}")
    log(f"  Output prefix:       {cfg.output_prefix}")

    t0 = time.time()
    engine = IterativeTypeAbstraction(cfg)
    engine.load(args.input)
    engine.run()
    engine._log_history_table()
    engine.save_final()

    total = time.time() - t0
    log(f"\nTotal wall time: {total:.1f}s")
    progress(f"Done in {total:.1f}s  |  see {log_path}")

    _log.close()


if __name__ == "__main__":
    main()