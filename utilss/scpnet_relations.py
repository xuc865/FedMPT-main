"""SCPNet label-relation matrices (semantic prior graph)."""
import os
import numpy as np


def build_default_relation(num_classes, seed=42):
    """
    Fallback relation matrix when relation+*.npy is missing.
    Uniform off-diagonal co-occurrence prior (diagonal cleared in SCPNet).
    Replace by running scripts/build_scpnet_relations.py on your training set.
    """
    rng = np.random.default_rng(seed)
    rel = rng.uniform(0.05, 0.15, size=(num_classes, num_classes)).astype(np.float32)
    np.fill_diagonal(rel, 0.0)
    return rel


def ensure_relation_matrix(path, num_classes):
    """Return path to .npy relation file, creating a default matrix if needed."""
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rel = build_default_relation(num_classes)
    np.save(path, rel)
    print(
        f"[SCPNet] Created default relation matrix at {path} "
        f"({num_classes}x{num_classes}). For paper-faithful results, "
        f"replace with scripts/build_scpnet_relations.py output."
    )
    return path
