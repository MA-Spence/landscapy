# Landscapy cookbook

The cookbook combines worked examples with the technical contracts for the
[0.9 public API](foundations/public-api.md). Start with installation, then use
recipe pages that define their input schema, check alignment and graph
invariants, interpret the output, and list common failure modes. Reference pages
live beside the recipes they govern, and each section index identifies them
separately.

New users should start with [Tutorial: constructing and analysing an
SSN](tutorial-constructing-and-analysing-an-ssn.md). It is a manually run,
end-to-end workflow from an unaligned FASTA and taxonomy CSV through PLM kNN,
TDA, evolutionary diffusion, annotation queries, category diffusion, Louvain
communities, and a quotient graph. Continue in the same Python session with
[Tutorial: Quantitative analysis of an
SSN](tutorial-quantitative-analysis-of-an-ssn.md) to attach replicated synthetic
fitness, select a scalar fitness view, and interpret graphical, spectral,
subsampling, and permutation analyses. Then continue with [Tutorial: ML
training and inference on an
SSN](tutorial-ml-training-and-inference-on-an-ssn.md) to mask held-out values,
train the same MLP on OHE and PLM features, wrap a user-supplied PyTorch model
with the `landscapy-ml` adapter interfaces, attach prediction layers, and
evaluate held-out rankings with Spearman's rho in figures.

## Sections

- [Installation, system requirements, optional features, and CI coverage](installation/README.md)
- [Foundations: empirical data and the landscape data model](foundations/README.md)
- [Components, graph topology, communities, and annotated groups](topology/README.md)
- [Graph construction and representation choice](graph-construction/README.md)
- [Saving, sharing, CLI use, and external visualization](io/README.md)
- [Ruggedness, autocorrelation, and spectral analysis](ruggedness/README.md)
- [Adaptive walks, accessibility, basins, optima, and neutral networks](accessibility/README.md)
- [Epistasis on complete, sampled, and categorical landscapes](epistasis/README.md)
- [Statistical inference and robustness analysis](statistics/README.md)
- [Simulation models and known-answer validation](simulation/README.md)
- [Validated exports for downstream machine learning](ml/README.md)
- [Scaling, backend selection, and reproducible execution](scaling/README.md)

## Shared example data

Recipes use the versioned [synthetic binary landscape](data/README.md). It is
small enough to audit by hand and deliberately carries no biological claim.
Stochastic recipes use fixed seeds. Examples assume the repository root is the
working directory when run from a checkout.

## What a recipe establishes

A successful example establishes that the inputs satisfy the stated software
contract and that the reported quantity was computed. It does not establish
that the chosen graph is biologically correct or that an estimator is evidence
for a biological mechanism. Representation choice, empirical sampling, and
component support remain part of the scientific model.

All fenced blocks beginning with `# cookbook: test` are executed by
`python scripts/check_cookbook_examples.py` in CI.
