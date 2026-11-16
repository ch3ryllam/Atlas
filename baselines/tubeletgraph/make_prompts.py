#!/usr/bin/env python3
"""
Build TubeletGraph/FluxGraph prompt masks from the frame-0 CropFormer entities, so both
baselines are prompted with every first-frame entity.

With --rev the prompts come from the last frame instead of the first, for the reverse
baseline: the reversed video starts there, so the mask is still written as frame 0.

Usage:
python3 baselines/tubeletgraph/make_prompts.py -d ours -s val
python3 baselines/tubeletgraph/make_prompts.py -d ours -s val --rev
"""

import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, rle_to_bmask


def ensure_entities(args, entity_dir, videos):
    """ Segment the videos that have no entities yet. """
    missing = [v for v in videos if not osp.exists(osp.join(entity_dir, f"{v}.json"))]
    if not missing:
        return
    print(f"No entities for {len(missing)} of {len(videos)} video(s); running CropFormer first.")
    subprocess.run(
        ["python3", "Atlas/entity_segmentation/cropformer.py",
         "-c", args.config, "-d", args.dataset, "-s", args.split],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default="configs/default.yaml")
    p.add_argument('-d', '--dataset', default="ours")
    p.add_argument('-s', '--split', default="val")
    p.add_argument('-o', '--out_dir', default=None, help="default: <intermdir>/tubeletgraph_prompts_<dataset>[_rev]")
    p.add_argument('--rev', action='store_true', help="prompt from the last frame, for the reverse baseline")
    args = p.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    entity_dir = osp.join(cfg.paths.intermdir, f"entities_{args.dataset}_{cfg.tubelet.entity_method}")
    suffix = "_rev" if args.rev else ""
    out_dir = args.out_dir or osp.join(cfg.paths.intermdir, f"tubeletgraph_prompts_{args.dataset}{suffix}")

    with open(osp.join(data_cfg.split_dir, f"{args.split}.txt")) as f:
        videos = [x.strip() for x in f if x.strip()]

    ensure_entities(args, entity_dir, videos)

    for video in videos:
        # One entry per frame, already thresholded and non-overlapping
        all_entities = json.load(open(osp.join(entity_dir, f"{video}.json")))
        frame_key = str(len(all_entities) - 1) if args.rev else "0"
        masks = [rle_to_bmask(rle) for rle in all_entities[frame_key].values()]
        assert len(masks) < 255, f"{video}: {len(masks)} entities do not fit in an indexed PNG"

        # Indexed PNG: 0 = background, 1..N = entities (the format TubeletGraph's load_anno expects)
        frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, video, data_cfg.image_format)))
        W, H = Image.open(frame_paths[-1 if args.rev else 0]).size
        prompt = np.zeros((H, W), dtype=np.uint8)
        for i, m in enumerate(masks):
            prompt[m] = i + 1
        os.makedirs(osp.join(out_dir, video), exist_ok=True)
        Image.fromarray(prompt).convert("P").save(osp.join(out_dir, video, "00000.png"))
        print(f"{video}: {len(masks)} entities at frame {frame_key}")

    print(f"Saved {len(videos)} prompt masks → {out_dir}")


if __name__ == "__main__":
    main()
