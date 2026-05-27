#!/usr/bin/env python3
"""
Build SCPNet relation+*.npy from label co-occurrence on a dataset split.
Example:
  python scripts/build_scpnet_relations.py --dataset voc --root ./data
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilss.paths import scp_relation_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["voc", "coco", "nus"])
    parser.add_argument("--root", type=str, default="./data")
    args = parser.parse_args()

    # Lazy import to avoid pulling torch until needed
    from config.defaults import _C as cfg_default
    from config.utils import reset_cfg
    from argparse import Namespace
    from convclip import clip
    from dataloader.dm_federated import TrainDataManager

    ns = Namespace(
        exp_name="build_rel",
        dataset=args.dataset,
        root=args.root,
        num_shots=8,
        depth_ctx=1,
        n_ctx=4,
        model_depth=0,
        num_epoch=1,
        batch_size=32,
        num_cls_per_client=1,
        avail_percent=1.0,
        pa=0.0,
        num_clusters=2,
        lr=0.001,
        temp=1,
        cond=4,
        lat=64,
        stun=0,
        cls=4,
        output_dir="./outputs",
        resume="",
        seed=34,
        backbone="ViT-B/16",
        saving=False,
        allow_resume=False,
        zsl=None,
        neg=2,
        pos=1,
        neda=False,
    )
    cfg = cfg_default.clone()
    reset_cfg(cfg, ns)
    cfg.freeze()

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device="cpu", jit=False)
    dm = TrainDataManager(cfg, args.dataset, clip_model, available_cls=list(range(cfg.TRAINER.ML.NUM_CLUSTERS)))
    nc = cfg.DATASET.NC
    cooc = np.zeros((nc, nc), dtype=np.float64)
    count = 0
    for _, labels, _ in dm.train_loader:
        for lab in labels:
            active = (lab > 0).nonzero(as_tuple=False).view(-1).tolist()
            for i in active:
                for j in active:
                    if i != j:
                        cooc[i, j] += 1
            count += 1
    if count == 0:
        raise RuntimeError("No training samples found; check --root and dataset layout.")
    rel = (cooc / max(count, 1)).astype(np.float32)
    out = scp_relation_path(args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, rel)
    print(f"Saved {out} shape={rel.shape} from {count} images.")


if __name__ == "__main__":
    main()
