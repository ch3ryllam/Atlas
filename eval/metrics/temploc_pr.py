"""
Temporal localization precision/recall (T_P, T_R) for transformation events.

A predicted event matches a GT transformation when its frame falls within the GT
interval extended by pad * round(fps * 0.5) frames on each side and the event types
(decomposition/composition) agree. Matching is one-to-one via Hungarian assignment;
T_P is the fraction of predictions matched, T_R the fraction of GT transformations
matched.

Usage:
python3 eval/metrics/temploc_pr.py -p _pred_out/ours -v <video> --pad 1 --json_out out.json --match_out matches.json
"""

import argparse
import json
import os.path as osp
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))
from utils import load_yaml_file


def add_count(counts, event_type, metric):
    counts["overall"][metric] += 1
    if event_type in counts:
        counts[event_type][metric] += 1


def compute_pr_by_type(pred_info, gt_info, pad=0, fps=15):
    """Match predicted event frames to GT temporal windows."""
    pad_frames = pad * round(fps * 0.5)
    counts = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in ["overall", "decomposition", "composition"]}
    matches = {}

    n_gt = len(gt_info)
    n_pred = len(pred_info)

    if n_gt == 0:
        for pred in pred_info:
            add_count(counts, pred.get("type", "unknown"), "fp")
        return counts, matches

    if n_pred == 0:
        for gt in gt_info:
            add_count(counts, gt.get("type", "unknown"), "fn")
        return counts, matches

    # A zero cost means the predicted frame falls inside the GT window and the event types agree
    cost = np.ones((n_gt, n_pred), dtype=np.float32)

    for i, gt in enumerate(gt_info):
        left = gt["start"] - pad_frames
        right = gt["end"] + pad_frames
        for j, pred in enumerate(pred_info):
            if left <= pred["frame"] <= right and pred.get("type") == gt.get("type"):
                cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_gts = set()
    matched_preds = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] != 0.0:
            continue

        matched_gts.add(i)
        matched_preds.add(j)

        gt_type = gt_info[i].get("type", "unknown")
        add_count(counts, gt_type, "tp")

        matches[str(i)] = pred_info[j]["frame"]

    for i, gt in enumerate(gt_info):
        if i not in matched_gts:
            add_count(counts, gt.get("type", "unknown"), "fn")

    for j, pred in enumerate(pred_info):
        if j not in matched_preds:
            add_count(counts, pred.get("type", "unknown"), "fp")

    return counts, matches


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-p", "--pred", type=str, required=True)
    parser.add_argument("-a", "--anno", type=str, default=None)
    parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--json_out", type=str, default=None)
    parser.add_argument("--match_out", type=str, default=None)
    return parser


def main():
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = cfg.datasets.ours
    anno_dir = args.anno or data_cfg.anno_dir
    fps = args.fps if args.fps is not None else data_cfg.fps

    with open(osp.join(anno_dir, f"{args.video}_transformations.json"), "r") as f:
        transformations = json.load(f)
    with open(osp.join(args.pred, f"{args.video}_events.json"), "r") as f:
        events = json.load(f)

    gt_info = [{"start": tf["start_frame"], "end": tf["end_frame"], "type": tf.get("type", "unknown")} for tf in transformations]
    pred_info = [{"frame": ev["frame"], "type": ev.get("type", "unknown")} for ev in events if "frame" in ev and ev["frame"] > 0]

    counts, matches = compute_pr_by_type(pred_info=pred_info, gt_info=gt_info, pad=args.pad, fps=fps)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(counts, f, indent=4)

    if args.match_out:
        with open(args.match_out, "w") as f:
            json.dump(matches, f, indent=4)


if __name__ == "__main__":
    main()
