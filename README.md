# Landscapy

Landscapy is a Python package for constructing and analysing graph-based fitness landscapes. It supports sequence-aware graph construction, layered fitness and annotation data, landscape models and ruggedness and epistasis analyses, with use cases in sequence similarity network analysis and protein engineering. Landscapy supports a generic interface to deep learning packages, maintained in [landscapy-ml](https://github.com/MA-Spence/landscapy-ml).

## Requirements and installation

Landscapy supports Python 3.11 and 3.12.

The default installation includes every user-facing feature in the `all` and `ml` dependency groups, together with a compatible 1.x release of `landscapy-ml`:

```bash
python -m pip install landscapy
```

For development from a checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

See [Installation and system requirements](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/installation/README.md) for more information on requirements and installation.

## Command-line interface and quickstart

```bash
landscapy --help
```

Given an aligned protein FASTA file (`sequences.fasta`), a minimal portable kNN CLI workflow is:

```bash
landscapy knn-landscape \
  --sequences sequences.fasta \
  --output landscape.pkl \
  --k 5 \
  --backend balltree \
  --embedding-domain ohe
```

The equivalent Python code is:

```python
from pathlib import Path

from fitness_landscape import FitnessLandscape, create_knn_graph
from fitness_landscape.core.graph import _encode_multiallele
from fitness_landscape.utils import fasta_to_prot20_sequences

sequences, aligned_sequences = fasta_to_prot20_sequences(
    "sequences.fasta",
    strict=False,
    return_gapped=True,
)
embedding_sequences = (
    aligned_sequences if aligned_sequences is not None else sequences
)
embeddings, _ = _encode_multiallele(embedding_sequences)

graph = create_knn_graph(
    sequences=sequences,
    k=5,
    embeddings=embeddings,
    embedding_domain="ohe",
    backend="balltree",
    _compute_hamming_edges=True,
)
landscape = FitnessLandscape.from_graph(
    graph,
    embeddings={"ohe": embeddings},
    active_embedding_domain="ohe",
)
landscape.save(Path("landscape.pkl"))
```

## Cookbook

The [worked-example cookbook](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/README.md) provides examples for common Landscapy usage. Every recipe states its assumptions, expected outputs, and limits of interpretation. Short recipe examples are checked in CI; the networked end-to-end tutorial is run manually.

- [Tutorial: constructing and analysing an SSN](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/tutorial-constructing-and-analysing-an-ssn.md)
- [Tutorial: Quantitative analysis of an SSN](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/tutorial-quantitative-analysis-of-an-ssn.md)
- [Tutorial: ML training and inference on an SSN](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/tutorial-ml-training-and-inference-on-an-ssn.md)
- [Installation and system requirements](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/installation/README.md)
- [Foundations](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/foundations/README.md)
- [Components, topology, and annotated groups](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/topology/README.md)
- [Graph construction and representation choice](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/graph-construction/README.md)
- [Saving, sharing, CLI use, and external visualization](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/io/README.md)
- [Ruggedness, autocorrelation, and spectral analysis](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/ruggedness/README.md)
- [Adaptive walks, accessibility, basins, optima, and neutral networks](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/accessibility/README.md)
- [Epistasis on complete, sampled, and categorical landscapes](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/epistasis/README.md)
- [Statistical inference and robustness analysis](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/statistics/README.md)
- [Simulation models and known-answer validation](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/simulation/README.md)
- [Validated exports for downstream machine learning](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/ml/README.md)
- [Scaling, backend selection, and reproducible execution](https://github.com/MA-Spence/landscapy/blob/main/docs/cookbook/scaling/README.md)

## License

Landscapy is distributed under the MIT License. See [LICENSE](LICENSE).
