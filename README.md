# Type Abstraction - structural typing for RDF entities

**License Notice**
Unless otherwise stated, the source code in this repository is licensed under the MIT License. The MIT License does not apply to third-party data or other third-party material contained in this repository. For details, see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

**Type Abstraction**

| Phase | Features |
|---|---|
| 0 (baseline) | `OUT:<pred>`, `IN:<pred>` - presence of outgoing/incoming predicates |
| N > 0 (refinement) | `OUT:p\|NR_TYPE:CID`, `IN:p\|NR_TYPE:CID` - the *inferred type* of the neighbour reached via `p` |

Similarity is **IDF-weighted cosine** over those features, so predicates that every entity
carries (IDF = 0) contribute nothing. Two entities are joined by an edge when their
similarity ≥ `--threshold`; each connected component of size ≥ `--min-size` whose members
share ≥ 1 informative feature becomes a type. Iteration stops when the partition is
stable (co-membership change ≤ `--epsilon`) or after `--max-iter` phases. One `rdf:type`
triple is emitted per clustered entity.

# Reproducibility
Everything below is copy-paste. Run from the repository root unless stated otherwise.

Original results are already in [reproducibility/sweeps](reproducibility/sweeps) and [reproducibility/results/watdiv_100k](reproducibility/results/watdiv_100k).

---

## 0. Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu 22.04, x86-64) |
| Python | 3.12 |

## 1. Environment

**Conda (recommended, tested)**. Any Python 3.12 environment with the dependencies works, 
but only the Conda path is verified.
 
**Conda**:
 
```bash
conda env create -f environment.yml
conda activate ta
```
 
Verify:
 
```bash
python -c "import rdflib, sklearn, matplotlib; print('ok')"
```
 
## 2. Type Abstraction Modules

| Module | Purpose |
|---|---|
| `type_abstractor.py` | the algorithm + CLI |
| `sweep_supervised.py` | **supervised** validation: strips `rdf:type`, sweeps the threshold, scores each inferred partition against the truth partition with ARI / AMI / V-measure |
| `sweep_unsupervised.py` | **unsupervised** diagnostics: strips `rdf:type`, cohesion, inter-cluster similarity, separation |

Uncovered entities get a singleton predicted label in the sweeps, so low coverage is
penalised rather than hidden.


### type_abstractor.py

```bash
# run
python -m type_abstraction.type_abstractor \
    --threshold 0.55 --min-size 2 --max-iter 5 --epsilon 0.01 datasets/watdiv_100k.nt
```

| Flag | Default | Meaning |
|---|---|---|
| `--threshold, -t` | 0.5 | similarity threshold for the co-type edge |
| `--min-size` | 2 | minimum cluster size to become a type |
| `--max-iter, -n` | 5 | maximum refinement phases |
| `--epsilon, -e` | 0.01 | convergence tolerance |
| `--output, -o` | `<dataset>_t<threshold>` | output directory (relative to the CWD) |
| `--save-all` | off | write NT + report for every phase, not just the final one |

### sweep_unsupervised.py (internal cluster validation)
```bash
# evaluate type abstracted clusters across different similarity thresholds (Silhouette, DB index, CH index)
python -m type_abstraction.sweep_unsupervised datasets/watdiv_100k.nt
```

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `<dataset>_unsup.csv` | output file name |
| `--min` | 0.1 | lowest similarity threshold to try |
| `--max` | 0.9 | highest similarity threshold to tr |
| `--step` | 0.05 | theshold increase with each consecutive sweep |
| `--max-iter` | 5 |  maximum refinement phases |


### sweep_supervised.py (external cluster validation)
```bash
# compare type abstracted clusters across different similarity thresholds with ground-truth types (ARI, AMI, V-Measure)
python -m type_abstraction.sweep_supervised datasets/watdiv_100k.nt --truth-mode iri
```
| Flag | Default | Meaning |
|---|---|---|
| `--truth-mode` | `type` | source of ground truth; `type` = ground truth from rdf:type triples or `iri`: truth from IRI local name minus trailing digits |
| `--truth-file` | None | separate file holding the ground-truth rdf:type triples |
| `--out` | `<dataset>_unsup.csv` | output file name |
| `--min` | 0.1 | lowest similarity threshold to try |
| `--max` | 0.9 | highest similarity threshold to tr |
| `--step` | 0.05 | theshold increase with each consecutive sweep |
| `--max-iter` | 5 |  maximum refinement phases |
