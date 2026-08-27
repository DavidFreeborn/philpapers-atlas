from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import robustness_semantic as semantic  # noqa: E402


def test_without_self_preserves_neighbor_order() -> None:
    indices = np.array([[2, 0, 1, 3], [1, 0, 2, 3]], dtype=np.int32)
    queries = np.array([0, 1], dtype=np.int32)
    result = semantic.without_self(indices, queries, 3)
    np.testing.assert_array_equal(result, np.array([[2, 1, 3], [0, 2, 3]]))


def test_label_permutation_null_preserves_cluster_sizes_and_is_reproducible() -> None:
    labels = np.array([0, 0, 0, 1, 1, -1, -1, 2])
    validation = np.array([0, 2, 4, 7])
    neighbors = np.array([[1, 2], [0, 3], [3, 5], [6, 5]])
    left = semantic.permutation_null(labels, validation, neighbors, repeats=5, seed=17)
    right = semantic.permutation_null(labels, validation, neighbors, repeats=5, seed=17)
    assert left == right
    assert len(left) == 5
    assert all(0 <= value <= 1 for value in left)


def test_npmi_rewards_consistent_cooccurrence() -> None:
    coherent = sparse.csr_matrix(np.array([[1, 1], [1, 1], [0, 0], [0, 0]]))
    separated = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))
    terms = np.array([0, 1])
    assert semantic.npmi_for_terms(coherent, terms) > semantic.npmi_for_terms(separated, terms)
