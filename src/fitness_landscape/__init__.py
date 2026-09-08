"""Construct and analyse graph-based fitness landscapes."""

from importlib import import_module

from . import core
from . import io

from .core import (
    FitnessLandscape,
    BaseNumpySequence,
    BinarySequence,
    MultialleleSequence,
    generate_sequences,
    sequence_distance,
    create_hamming_graph,
    create_knn_graph,
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
)
from .io import export_lsbundle, load_bundle_dir, save_bundle_dir

__version__ = '1.0.0'

__all__ = [
    'core',
    'transforms',
    'analysis',
    'models',
    'io',
    'BaseNumpySequence',
    'BinarySequence',
    'MultialleleSequence',
    'generate_sequences',
    'sequence_distance',
    'FitnessLandscape',
    'create_hamming_graph',
    'create_knn_graph',
    'NumericFitness',
    'CategoricalFitness',
    'ProbabilisticCategoricalFitness',
    'save_bundle_dir',
    'load_bundle_dir',
    'export_lsbundle',
]

_LAZY_SUBMODULES = {"analysis", "models", "phylo", "transforms"}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
