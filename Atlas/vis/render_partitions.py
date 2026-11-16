"""
Render a 2x2 debug video of the two tubelet partitions: all forward and all reverse
tubelets on the top row, the emergent ones on the bottom row. Reads the tubelets only,
so it runs before any event detection or prompting.

Usage: python3 Atlas/vis/render_partitions.py -c configs/default.yaml -d ours -i crackers
"""

import argparse
import concurrent.futures
import glob
import os
import os.path as osp
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, load_tracks, write_video
from Atlas.vis.render_utils import overlay


def resize_img(img, w, h):
    return cv2.resize(img, (w, h))


def get_emergent_ids(fwd_meta, rev_meta, num_frames):
    """ Ids of the tubelets that emerge mid-video: after the first frame going forward,
    before the last frame going in reverse (reverse tracks are in forward frame indices).
    """
    fwd_ids = {oid for oid, info in fwd_meta.items() if min(info["frames"]) != 0}
    rev_ids = {oid for oid, info in rev_meta.items() if max(info["frames"]) != num_frames - 1}
    return fwd_ids, rev_ids


def select_masks(tracks, idx, prefix, keep_ids=None):
    """ Masks at frame idx, keyed by direction-prefixed id so the two partitions get
    separate colors. Restricted to keep_ids when given.
    """
    return {
        f"{prefix}_{oid}": rle
        for oid, rle in tracks.get(idx, {}).items()
        if keep_ids is None or oid in keep_ids
    }


def render_single_frame(task):
    f_path, nw, nh, all_fwd, all_rev, emg_fwd, emg_rev = task

    base = np.array(Image.open(f_path), dtype=np.uint8)
    panels = [
        resize_img(overlay(base, masks), nw, nh)
        for masks in (all_fwd, all_rev, emg_fwd, emg_rev)
    ]
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])


def generate_partition_video(cfg, args, frame_paths, fwd_tracks, rev_tracks, fwd_meta, rev_meta, fps):
    with Image.open(frame_paths[0]) as img:
        w, h = img.size
    new_w, new_h = 640, int(h * (640 / w))
    fwd_ids, rev_ids = get_emergent_ids(fwd_meta, rev_meta, len(frame_paths))

    render_tasks = []
    for idx in range(len(frame_paths)):
        render_tasks.append((
            frame_paths[idx],
            new_w,
            new_h,
            select_masks(fwd_tracks, idx, "F"),
            select_masks(rev_tracks, idx, "R"),
            select_masks(fwd_tracks, idx, "F", fwd_ids),
            select_masks(rev_tracks, idx, "R", rev_ids),
        ))

    vis_out = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(render_single_frame, render_tasks)
        for result in tqdm(results, total=len(render_tasks), desc="Rendering Frames"):
            vis_out.append(result)

    out_path = osp.join(cfg.paths.visdir, args.dataset, f"{args.instance}_partitions.mp4")
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    write_video(out_path, vis_out, fps=fps)
    print(f"Partition video saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-d", "--dataset", default="vost")
    parser.add_argument("-i", "--instance", required=True)
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, args.instance, data_cfg.image_format)))
    tracks_dir = osp.join(cfg.paths.intermdir, f"tubelets_{args.dataset}_{cfg.tubelet.entity_method}")

    fwd_tracks, fwd_meta = load_tracks(tracks_dir, args.instance, is_rev_dir=False)
    rev_tracks, rev_meta = load_tracks(tracks_dir, args.instance, is_rev_dir=True, num_frames=len(frame_paths))

    generate_partition_video(
        cfg, args, frame_paths, fwd_tracks, rev_tracks, fwd_meta, rev_meta, data_cfg.fps
    )


if __name__ == "__main__":
    main()
