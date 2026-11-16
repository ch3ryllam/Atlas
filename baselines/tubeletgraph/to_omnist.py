#!/usr/bin/env python3
"""
Convert TubeletGraph or FluxGraph predictions (one file per prompt object, after its VLM step) to the
per-video format that eval/eval_omnist.py reads. Both write the same per-object files, so the only
difference is --in_dir. Follows the forward TubeletGraph baseline (TubeletGraph_Baseline_F): tracks are
{video}_{obj}_{piece} (piece 0 = prompt object), every kept piece is a decomposition of its prompt
object at the frame it starts, and there are no forward/reverse pairs.

Usage:
python3 baselines/tubeletgraph/to_omnist.py --in_dir thirdparty/TubeletGraph/_pred_out/ours-val-Ours_gpt-4.1 --out_dir _pred_out_baselines/TubeletGraph
python3 baselines/tubeletgraph/to_omnist.py --in_dir thirdparty/FluxGraph/_pred_out/ours-val-Ours_gpt-4.1 --out_dir _pred_out_baselines/FluxGraph
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file


def convert_video(video, in_dir, rev=False):
    """Return (events, final_masks) for one video from its per-object files.

    A reverse run saw the clip backwards, so its frame indices are mapped back to the
    original video and each piece is a composition rather than a decomposition.
    """
    obj_paths = [p for p in glob.glob(osp.join(in_dir, glob.escape(video) + "_*.json"))
                 if osp.basename(p)[len(video) + 1:-len(".json")].isdigit()]
    obj_paths.sort(key=lambda p: int(osp.basename(p)[len(video) + 1:-len(".json")]))
    datas = {osp.basename(p)[len(video) + 1:-len(".json")]: json.load(open(p)) for p in obj_paths}

    # The prompt object is tracked over the whole clip, so the highest frame gives its length
    last_frame = max((int(f) for data in datas.values() for f in data["supix_masks"]), default=0)
    to_orig = (lambda f: last_frame - f) if rev else (lambda f: f)

    events, final_masks = [], {}
    for obj, data in datas.items():
        obj_info = data.get("obj_info", {})

        # Every mask the method keeps for this prompt object, per piece (VLM-rejected pieces are already removed)
        for frame, pieces in data["supix_masks"].items():
            for piece, rle in pieces.items():
                final_masks.setdefault(f"{video}_{obj}_{piece}", {})[str(to_orig(int(frame)))] = rle

        # Each kept piece separates from the prompt object at the frame where the piece starts,
        # which read in forward time is a composition for a reverse run
        parent, child_of = f"{video}_{obj}_0", f"{video}_{obj}_{{}}"
        for piece, info in obj_info.items():
            if piece == "0" or "object_start_frame_idx" not in info:
                continue
            child = child_of.format(piece)
            events.append({
                "frame": to_orig(int(info["object_start_frame_idx"])),
                "type": "composition" if rev else "decomposition",
                "start_state_objects": [parent, child] if rev else [parent],
                "end_state_objects": [parent] if rev else [parent, child],
                "vlm_action": info.get("action", ""),
                "vlm_parent_name": info.get("prior_desc") or obj_info.get("0", {}).get("desc", ""),
                "vlm_child_name": info.get("desc", ""),
            })
    events.sort(key=lambda ev: ev["frame"])
    return events, final_masks, len(obj_paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', default="thirdparty/FluxGraph/_pred_out/ours-val-Ours_gpt-4.1", help="TubeletGraph or FluxGraph predictions after its VLM step")
    p.add_argument('--out_dir', default="_pred_out_baselines/FluxGraph")
    p.add_argument('-c', '--config', default="configs/default.yaml")
    p.add_argument('-d', '--dataset', default="ours")
    p.add_argument('-s', '--split', default="val", help="split name in the dataset's split_dir")
    p.add_argument('--rev', action='store_true', help="the run saw the clip reversed, so events are compositions")
    args = p.parse_args()

    data_cfg = getattr(load_yaml_file(args.config).datasets, args.dataset)
    with open(osp.join(data_cfg.split_dir, f"{args.split}.txt")) as f:
        videos = [x.strip() for x in f if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    missing = []
    for video in videos:
        in_dir = args.in_dir.format(video=video) if "{video}" in args.in_dir else args.in_dir
        events, final_masks, num_objs = convert_video(video, in_dir, rev=args.rev)
        if num_objs == 0:
            missing.append(video)
            continue
        for suffix, data in [("events", events), ("final_masks", final_masks), ("pairs", [])]:
            with open(osp.join(args.out_dir, f"{video}_{suffix}.json"), "w") as f:
                json.dump(data, f)
        print(f"{video}: {num_objs} objects, {len(events)} events")
    if missing:
        print(f"[!] No predictions for {len(missing)} video(s): {', '.join(missing)}")
    print(f"Saved → {args.out_dir}")


if __name__ == "__main__":
    main()
