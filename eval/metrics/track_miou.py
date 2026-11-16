"""
Tubelet tracking mIoU (J_tubelet) between GT objects and predicted final masks.

GT and predicted tracks are split into temporally stable tubelets at their respective
transformation boundaries, excluding frames inside annotated transformation intervals. 
Split tubelets are matched one-to-one via Hungarian assignment on whole-track mean IoU, 
averaged over the frames where the GT tubelet exists. J_tubelet is the summed matched 
IoU over the GT tubelet count, so unmatched GT tubelets count as zero.

Usage:
python3 eval/metrics/track_miou.py -i <video> --pred_dir _pred_out/ours --json_out out.json
"""

import argparse
import json
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))
from utils import load_yaml_file


def extract_base_id(obj):
    if isinstance(obj, dict):
        obj = obj.get("id", obj)
    return str(obj).split("_")[-1]


def split_tracks_around_events(gt_annotations, gt_transformations, pred_masks, pred_events, pred_pairs):
    """Split GT and predicted tracks into stable tubelets, excluding frames inside transformation zones."""
    def merge_zones(zones):
        if not zones:
            return []
        merged = [zones[0]]
        for start, end in zones[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    gt_zones = defaultdict(list)
    for tf in gt_transformations:
        for obj in tf.get("start_state_objects", []) + tf.get("end_state_objects", []):
            gt_zones[extract_base_id(obj)].append((tf["start_frame"], tf["end_frame"]))
    gt_zones = {k: merge_zones(sorted(v, key=lambda x: x[0])) for k, v in gt_zones.items()}

    pred_zones = defaultdict(list)
    for ev in pred_events:
        if "frame" in ev:
            for obj_id in ev.get("start_state_objects", []) + ev.get("end_state_objects", []):
                pred_zones[str(obj_id)].append((ev["frame"], ev["frame"]))
    pred_zones = {k: merge_zones(sorted(v, key=lambda x: x[0])) for k, v in pred_zones.items()}

    clean_gt = defaultdict(dict)
    for frame_str, objects in gt_annotations.items():
        frame_idx = int(frame_str)
        for obj_id, obj_data in objects.items():
            zones = gt_zones.get(extract_base_id(obj_id), [])
            if any(start <= frame_idx <= end for start, end in zones):
                continue
            clean_gt[f"{obj_id}_part{sum(1 for start, end in zones if frame_idx > end)}"][frame_str] = {"size": obj_data["size"], "counts": obj_data["counts"]}

    clean_preds = defaultdict(dict)
    for pred_track_id, frames in pred_masks.items():
        base_track_id = pred_track_id.split("_part")[0]
        if base_track_id.startswith("Pair_"):
            try:
                pair_idx = int(base_track_id.split("_")[1])
                source_obj_ids = [str(x) for x in pred_pairs[pair_idx]] if pair_idx < len(pred_pairs) else [base_track_id]
            except ValueError:
                source_obj_ids = [base_track_id]
        else:
            source_obj_ids = [base_track_id]

        zones = merge_zones(sorted([zone for source_obj_id in source_obj_ids for zone in pred_zones.get(source_obj_id, [])], key=lambda x: x[0]))
        for frame_str, mask_data in frames.items():
            frame_idx = int(frame_str)
            if any(start <= frame_idx <= end for start, end in zones):
                continue
            clean_preds[f"{pred_track_id}_part{sum(1 for start, end in zones if frame_idx > end)}"][frame_str] = mask_data

    return clean_gt, clean_preds


def compute_tracking_miou(
    gt_annotations,
    gt_transformations,
    pred_masks,
    pred_events,
    pred_pairs,
):
    """Compute average matched-track IoU for all objects and by event type."""
    clean_gt, clean_preds = split_tracks_around_events(gt_annotations, gt_transformations, pred_masks, pred_events, pred_pairs)

    if not clean_gt:
        return {"error": "Missing ground truth."}

    base_id_to_types = defaultdict(set)
    # Map each GT object to the transformation types it participates in
    for tf in gt_transformations:
        for obj in tf.get("start_state_objects", []) + tf.get("end_state_objects", []):
            base_id_to_types[extract_base_id(obj)].add(tf.get("type", "unknown"))

    stats = {cat: {"iou_sum": 0.0, "gt_count": 0} for cat in ["overall", "decomposition", "composition"]}

    for gt_id in clean_gt.keys():
        base_id = extract_base_id(gt_id.split("_part")[0])
        stats["overall"]["gt_count"] += 1
        for t in base_id_to_types.get(base_id, set()):
            if t in stats:
                stats[t]["gt_count"] += 1

    # Match each GT object to at most one predicted object by maximum whole-track mean IoU
    gt_ids, pred_ids = list(clean_gt.keys()), list(clean_preds.keys())
    if gt_ids and pred_ids:
        iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
        for i, gt_id in enumerate(gt_ids):
            gt_track = clean_gt[gt_id]
            for j, pred_id in enumerate(pred_ids):
                pred_track = clean_preds[pred_id]
                iou_matrix[i, j] = sum(float(mask_utils.iou([pred_track[f]], [gt_track[f]], [0])[0][0]) for f in gt_track if f in pred_track) / len(gt_track)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        for i, j in zip(row_ind, col_ind):
            base_id = extract_base_id(gt_ids[i].split("_part")[0])
            stats["overall"]["iou_sum"] += float(iou_matrix[i, j])
            for t in base_id_to_types.get(base_id, set()):
                if t in stats:
                    stats[t]["iou_sum"] += float(iou_matrix[i, j])

    return {"stats": stats}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--instance", type=str, required=True)
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--anno_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default="_pred_out")
    parser.add_argument("--json_out", type=str, default=None)
    return parser


def main():
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    anno_dir = args.anno_dir or cfg.datasets.ours.anno_dir

    with open(osp.join(anno_dir, f"{args.instance}_annotations.json"), "r") as f:
        gt_annotations = json.load(f)
    with open(osp.join(anno_dir, f"{args.instance}_transformations.json"), "r") as f:
        gt_transformations = json.load(f)
    with open(osp.join(args.pred_dir, f"{args.instance}_final_masks.json"), "r") as f:
        pred_masks = json.load(f)
    with open(osp.join(args.pred_dir, f"{args.instance}_events.json"), "r") as f:
        pred_events = json.load(f)
    with open(osp.join(args.pred_dir, f"{args.instance}_pairs.json"), "r") as f:
        pred_pairs = json.load(f)

    if args.json_out is None:
        args.json_out = f"{args.instance}_track_stats.json"

    results = compute_tracking_miou(
        gt_annotations=gt_annotations,
        gt_transformations=gt_transformations,
        pred_masks=pred_masks,
        pred_events=pred_events,
        pred_pairs=pred_pairs,
    )

    if args.json_out and "error" not in results:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
