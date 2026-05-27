"""Repository path helpers (replaces hard-coded PATHPDBB placeholders)."""
import os

# Repo root: .../FedMPT-main
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root():
    return REPO_ROOT


def scps_config_dir():
    return os.path.join(REPO_ROOT, "model", "scps")


def scp_relation_path(dataset_name):
    """Path to SCPNet label-relation matrix for a dataset."""
    fname = {
        "voc": "relation+voc.npy",
        "coco": "relation+coco.npy",
        "nus": "relation+nuswide.npy",
        "multiscene": "relation+multiscene.npy",
        "mlrsnet": "relation+mlrsnet.npy",
    }.get(dataset_name)
    if fname is None:
        raise ValueError(f"No SCPNet relation file mapping for dataset={dataset_name}")
    return os.path.join(scps_config_dir(), fname)


def labs_dir():
    """ZSL split JSONs (optional; only needed when --zsl is set)."""
    return os.path.join(REPO_ROOT, "labs")


def cluster_cache_dir(output_dir):
    """Federated client clustering cache (ViT-B/16 features + KMeans)."""
    return os.path.join(output_dir, "clusters")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
