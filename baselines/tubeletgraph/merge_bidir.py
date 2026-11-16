"""
Combine the forward and reverse TubeletGraph runs into the bidirectional baseline: event
objects whose tubelets overlap are merged into one Pair_* id, unmatched ones keep their own.
Produces the "TubeletGraph-Bidir" row.

Usage: python3 baselines/tubeletgraph/merge_bidir.py -d ours -s val
"""

import argparse
import json
import os.path as osp
import sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine forward/reverse TubeletGraph event objects. Forward/reverse "
            "objects with average tubelet IoU >= threshold are merged into the same "
            "Pair_* id; unmatched event objects get their own Pair_* id."
        )
    )
    parser.add_argument("--f_root", type=Path, default="_pred_out_baselines/TubeletGraph_Baseline_F")
    parser.add_argument("--r_root", type=Path, default="_pred_out_baselines/TubeletGraph_Baseline_R")
    parser.add_argument("--out_root", type=Path, default="_pred_out_baselines/TubeletGraph_Baseline_B")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-d", "--dataset", default="ours")
    parser.add_argument("-s", "--split", default="val", help="split name in the dataset's split_dir")
    parser.add_argument("--iou_thresh", type=float, default=0.75)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    data_cfg = getattr(load_yaml_file(args.config).datasets, args.dataset)
    with open(osp.join(data_cfg.split_dir, f"{args.split}.txt")) as f:
        videos = [line.strip() for line in f if line.strip()]

    for video in videos:
        f_events_path = args.f_root / f"{video}_events.json"
        r_events_path = args.r_root / f"{video}_events.json"
        f_masks_path = args.f_root / f"{video}_final_masks.json"
        r_masks_path = args.r_root / f"{video}_final_masks.json"

        out_events_path = args.out_root / f"{video}_events.json"
        out_masks_path = args.out_root / f"{video}_final_masks.json"
        out_pairs_path = args.out_root / f"{video}_pairs.json"

        f_events = json.load(f_events_path.open("r")) if f_events_path.exists() else []
        r_events = json.load(r_events_path.open("r")) if r_events_path.exists() else []
        f_raw_masks = json.load(f_masks_path.open("r")) if f_masks_path.exists() else {}
        r_raw_masks = json.load(r_masks_path.open("r")) if r_masks_path.exists() else {}

        # Only keep objects that appear in transformation events.
        f_event_obj_ids = set()
        r_event_obj_ids = set()

        for event in f_events:
            f_event_obj_ids.update(event.get("start_state_objects", []))
            f_event_obj_ids.update(event.get("end_state_objects", []))

        for event in r_events:
            r_event_obj_ids.update(event.get("start_state_objects", []))
            r_event_obj_ids.update(event.get("end_state_objects", []))

        f_masks = {
            f"F_{obj_id}": f_raw_masks[obj_id]
            for obj_id in sorted(f_event_obj_ids, key=str)
            if obj_id in f_raw_masks
        }

        r_masks = {
            f"R_{obj_id}": r_raw_masks[obj_id]
            for obj_id in sorted(r_event_obj_ids, key=str)
            if obj_id in r_raw_masks
        }

        # Score every F/R pair by average IoU over shared frames.
        f_ids, r_ids = list(f_masks), list(r_masks)
        iou_matrix = np.zeros((len(f_ids), len(r_ids)))

        for i, (f_obj, f_track) in enumerate(f_masks.items()):
            for j, (r_obj, r_track) in enumerate(r_masks.items()):
                shared_frames = set(f_track.keys()) & set(r_track.keys())
                if not shared_frames:
                    continue

                total_iou = 0.0

                for frame in shared_frames:
                    total_iou += float(
                        mask_utils.iou(
                            [f_track[frame]],
                            [r_track[frame]],
                            [False],
                        )[0][0]
                    )

                iou_matrix[i, j] = total_iou / len(shared_frames)

        # Hungarian matching: each F object is merged with at most one R object, keeping pairs >= threshold.
        matches = []
        if iou_matrix.size:
            rows, cols = linear_sum_assignment(iou_matrix, maximize=True)
            matches = [(iou_matrix[i, j], f_ids[i], r_ids[j]) for i, j in zip(rows, cols) if iou_matrix[i, j] >= args.iou_thresh]
        matches.sort(reverse=True, key=lambda x: x[0])

        obj_to_merged_id = {}
        pair_idx = 0

        for avg_iou, f_obj, r_obj in matches:
            merged_id = f"Pair_{pair_idx}"
            pair_idx += 1

            obj_to_merged_id[f_obj] = merged_id
            obj_to_merged_id[r_obj] = merged_id

        # Unmatched objects still get Pair_* ids.
        for obj_id in list(f_masks.keys()) + list(r_masks.keys()):
            if obj_id not in obj_to_merged_id:
                obj_to_merged_id[obj_id] = f"Pair_{pair_idx}"
                pair_idx += 1

        merged_masks = {}

        for obj_id, frames in {**f_masks, **r_masks}.items():
            merged_id = obj_to_merged_id[obj_id]
            merged_masks.setdefault(merged_id, {})

            # If an F/R pair shares the same Pair_* id and the same frame exists in
            # both tracks, keep the forward mask by default for deterministic output.
            for frame, rle in frames.items():
                if frame not in merged_masks[merged_id]:
                    merged_masks[merged_id][frame] = rle
                else:
                    existing = merged_masks[merged_id][frame]

                    existing_mask = mask_utils.decode(existing).astype(bool)
                    new_mask = mask_utils.decode(rle).astype(bool)

                    union_mask = np.logical_or(existing_mask, new_mask).astype(np.uint8)

                    merged_masks[merged_id][frame] = mask_utils.encode(
                        np.asfortranarray(union_mask)
                    )
                    merged_masks[merged_id][frame]["counts"] = (
                        merged_masks[merged_id][frame]["counts"].decode("utf-8")
                    )

        merged_events = []

        for event in f_events:
            event = dict(event)
            event["type"] = "decomposition"
            event["start_state_objects"] = [
                obj_to_merged_id[f"F_{obj_id}"]
                for obj_id in event.get("start_state_objects", [])
                if f"F_{obj_id}" in obj_to_merged_id
            ]
            event["end_state_objects"] = [
                obj_to_merged_id[f"F_{obj_id}"]
                for obj_id in event.get("end_state_objects", [])
                if f"F_{obj_id}" in obj_to_merged_id
            ]
            merged_events.append(event)

        for event in r_events:
            event = dict(event)
            event["type"] = "composition"
            event["start_state_objects"] = [
                obj_to_merged_id[f"R_{obj_id}"]
                for obj_id in event.get("start_state_objects", [])
                if f"R_{obj_id}" in obj_to_merged_id
            ]
            event["end_state_objects"] = [
                obj_to_merged_id[f"R_{obj_id}"]
                for obj_id in event.get("end_state_objects", [])
                if f"R_{obj_id}" in obj_to_merged_id
            ]
            merged_events.append(event)

        merged_events.sort(key=lambda e: (int(e["frame"]), e["type"]))

        with out_events_path.open("w") as f:
            json.dump(merged_events, f, indent=4)

        with out_masks_path.open("w") as f:
            json.dump(merged_masks, f, indent=4)

        with out_pairs_path.open("w") as f:
            json.dump([], f, indent=4)

        print(
            f"{video}: "
            f"{len(f_masks)} F event objs + {len(r_masks)} R event objs "
            f"-> {len(merged_masks)} Pair_* objs, "
            f"{len(matches)} IoU matches >= {args.iou_thresh}, "
            f"{len(merged_events)} events"
        )


if __name__ == "__main__":
    main()