#!/usr/bin/env python3
"""
Evaluate prompt-object tracking by "replaying" predicted state transitions from the
initial prompt objects and comparing the resulting masks with GT annotations.

For each prompt object, we start from its matched predicted object ID and update
that active ID set at every predicted event. For composition/assembly, a prompt
object that is the merging child disappears at the event, while a prompt object
that is the surviving parent continues as the merged parent. For decomposition,
the object expands to the union of the post-split objects, so the tracked prompt mask
follows both resulting parts.

By default this reads outputs from:
  <cfg.paths.outdir>/<dataset>/

Use --pred_dir to evaluate another method directly.

Usage:
python eval/eval_tracking.py -c configs/default.yaml -d vost -s val
python eval/eval_tracking.py -c configs/default.yaml -d vscos -s val
python eval/eval_tracking.py -c configs/default.yaml -d m3vos -s val
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
from pycocotools import mask as mask_utils

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from utils import load_yaml_file, rle_to_bmask


def replay_pred_tracks(seed_ids, seed_matches, pred_masks, pred_events, pred_pairs, frame_count):
    pred_tracks = {}

    pair_to_nodes = {}
    node_to_pair = {}
    for pair_idx, pair_nodes in enumerate(pred_pairs):
        pair_id = f"Pair_{pair_idx}"
        nodes = {str(node) for node in pair_nodes}
        pair_to_nodes[pair_id] = nodes
        for node in nodes:
            node_to_pair[node] = pair_id

    for seed_id in seed_ids:
        seed_match = seed_matches.get(seed_id)
        if seed_match is None:
            pred_tracks[seed_id] = {}
            continue

        matched_obj_id = seed_match
        active = set(pair_to_nodes.get(matched_obj_id, {matched_obj_id}))
        track = {}
        event_pos = 0

        for frame_idx in range(frame_count):
            while event_pos < len(pred_events) and int(pred_events[event_pos]["frame"]) == frame_idx:
                event = pred_events[event_pos]
                start_ids = set(str(x) for x in event.get("start_state_objects", []))
                end_ids = set(str(x) for x in event.get("end_state_objects", []))
                if active & start_ids:
                    if event["type"] == "composition":
                        active = (active - start_ids) | (active & end_ids)
                    else:
                        active = (active - start_ids) | end_ids
                event_pos += 1

            frame_str = str(frame_idx)
            mask_obj_ids = set()
            for node in active:
                if node in pred_masks:
                    mask_obj_ids.add(node)
                elif node in node_to_pair and node_to_pair[node] in pred_masks:
                    mask_obj_ids.add(node_to_pair[node])
            rles = [pred_masks[obj_id][frame_str] for obj_id in sorted(mask_obj_ids) if frame_str in pred_masks[obj_id]]
            rles = [{"size": r["size"], "counts": r["counts"].encode("ascii") if isinstance(r["counts"], str) else r["counts"]} for r in rles]
            if rles:
                merged = mask_utils.merge(rles, intersect=0)
                counts = merged["counts"].decode("ascii") if isinstance(merged["counts"], bytes) else merged["counts"]
                track[frame_str] = {"size": merged["size"], "counts": counts}

        pred_tracks[seed_id] = track

    return pred_tracks


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking by replaying state transitions")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-s", "--split", default="val")
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    split_path = osp.join(data_cfg.split_dir, f"{args.split}.txt")
    with open(split_path, "r") as f:
        instances = [line.strip() for line in f if line.strip()]
    if args.limit is not None:
        instances = instances[: args.limit]

    pred_dir = args.pred_dir or osp.join(cfg.paths.outdir, args.dataset)
    png_dir = osp.join(cfg.paths.evaldir, args.dataset, f"tracking_png_{args.split}")
    os.makedirs(png_dir, exist_ok=True)

    for idx, instance in enumerate(instances, start=1):
        print(f"[{idx}/{len(instances)}] Evaluating {instance}")
        with open(osp.join(pred_dir, f"{instance}_final_masks.json"), "r") as f:
            pred_masks = json.load(f)
        with open(osp.join(pred_dir, f"{instance}_events.json"), "r") as f:
            pred_events = json.load(f)
        with open(osp.join(pred_dir, f"{instance}_pairs.json"), "r") as f:
            pred_pairs = json.load(f)
        with open(osp.join(pred_dir, f"{instance}_prompt_final_map.json"), "r") as f:
            prompt_final_map = json.load(f)
        anno_paths = sorted(glob.glob(osp.join(data_cfg.anno_dir, instance, data_cfg.anno_format)))
        if not anno_paths:
            raise FileNotFoundError(f"No annotation PNGs found for {instance} in {data_cfg.anno_dir}")
        frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, instance, data_cfg.image_format)))
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found for {instance} in {data_cfg.image_dir}")
        seed_ids = [str(int(obj_id)) for obj_id in np.unique(np.array(Image.open(anno_paths[0]))) if obj_id not in (0, 255)]
        pred_events = sorted(pred_events, key=lambda x: (int(x.get("frame", -1)), x.get("type", "")))
        seed_matches = {str(seed_id): str(prompt_final_map[str(seed_id)]) for seed_id in seed_ids if str(seed_id) in prompt_final_map}
        pred_tracks = replay_pred_tracks(seed_ids, seed_matches, pred_masks, pred_events, pred_pairs, len(frame_paths))

        prompt_mask = Image.open(anno_paths[0])
        palette = prompt_mask.getpalette()
        frame_names = [osp.splitext(osp.basename(p))[0] + '.png' for p in frame_paths]
        png_masks = [None] * len(frame_names)

        for seed_id, track in pred_tracks.items():
            if not track:
                continue
            obj_idx = int(seed_id)
            for frame_str, rle in track.items():
                frame_idx = int(frame_str)
                bmask = rle_to_bmask(rle)
                if png_masks[frame_idx] is None:
                    png_masks[frame_idx] = np.zeros(bmask.shape, dtype=np.uint8)
                png_masks[frame_idx][bmask & (png_masks[frame_idx] == 0)] = obj_idx

        instance_png_dir = osp.join(png_dir, instance)
        os.makedirs(instance_png_dir, exist_ok=True)
        for frame_idx, frame_name in enumerate(frame_names):
            mask_img = png_masks[frame_idx]
            if mask_img is None:
                mask_img = np.zeros(np.array(prompt_mask).shape, dtype=np.uint8)
            out_img = Image.fromarray(mask_img)
            out_img.putpalette(palette)
            out_img.save(osp.join(instance_png_dir, frame_name))

    eval_set = args.split
    for candidate_set in ('train', 'val', 'test'):
        if args.split == candidate_set or args.split.startswith(f'{candidate_set}_'):
            eval_set = candidate_set
            break

    eval_cmd = ['python3', 'thirdparty/RMem/evaluation/evaluation_method.py', '--results_path', png_dir, '--dataset_path', data_cfg.data_dir, '--set', eval_set, '--re']
    if eval_set != args.split:
        eval_cmd.extend(['--sequences_file', split_path, '--output_set', args.split])
    subprocess.run(eval_cmd, check=True)

    print(f"Tracking PNGs saved to: {png_dir}")
    print(f"Official tracking CSV saved to: {osp.join(png_dir, f'global_results-{args.split}.csv')}")


if __name__ == "__main__":
    main()
