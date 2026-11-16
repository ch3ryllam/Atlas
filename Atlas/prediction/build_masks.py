"""
Build the final mask of every object in a video from its forward and reverse tubelets,
the matched pairs, and the transformation events the VLM reported.

Usage: python3 Atlas/prediction/build_masks.py -c configs/default.yaml -d ours -i crackers
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import bmask_to_rle, load_yaml_file, rle_to_bmask, load_tracks, save_timing
from Atlas.prediction.detect_events import union_pair_masks


def union_masks(masks):
    """ Union of the given masks, or None when none of them cover anything. """
    bmasks = [rle_to_bmask(mask) for mask in masks if mask is not None]
    if not bmasks:
        return None
    combined = np.logical_or.reduce(bmasks)
    return bmask_to_rle(combined) if combined.any() else None


def get_pair_masks(pair_valid_frames, pairs, trimmed_fwd, trimmed_rev):
    """ Mask of each pair on each frame it is valid on: the union of its forward and its
    reverse member, which are the same object tracked in the two directions. Frames where
    both were trimmed away are left out, so a pair's frames are the frames it exists on.
    """
    pair_masks = defaultdict(dict)
    for pair_idx, frames in pair_valid_frames.items():
        fwd_node, rev_node = pairs[pair_idx]
        fwd_id, rev_id = fwd_node.split("_")[1], rev_node.split("_")[1]
        for frame_idx in frames:
            fwd_mask = trimmed_fwd.get(frame_idx, {}).get(fwd_id)
            rev_mask = trimmed_rev.get(frame_idx, {}).get(rev_id)
            mask = union_masks([fwd_mask, rev_mask])
            if mask is not None:
                pair_masks[pair_idx][frame_idx] = mask

    return pair_masks


def load_final_mask_inputs(cfg, args, instance):
    """ Load what the final masks are built from: the frames, the forward and reverse
    tubelets, and the events and pairs the earlier stages of the pipeline wrote.
    """
    data_cfg = getattr(cfg.datasets, args.dataset)
    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, instance, data_cfg.image_format)))
    tracks_dir = osp.join(cfg.paths.intermdir, f"tubelets_{args.dataset}_{cfg.tubelet.entity_method}")
    fwd_tracks, fwd_meta = load_tracks(tracks_dir, instance, is_rev_dir=False)
    rev_tracks, rev_meta = load_tracks(tracks_dir, instance, is_rev_dir=True, num_frames=len(frame_paths))

    out_dir = osp.join(cfg.paths.outdir, get_output_dataset(args))
    with open(osp.join(out_dir, f"{instance}_events.json"), "r") as f:
        formatted_events = json.load(f)
    with open(osp.join(out_dir, f"{instance}_pairs.json"), "r") as f:
        pairs = [tuple(x) for x in json.load(f)]

    events = []
    for event in formatted_events:
        if event["type"] == "decomposition":
            parent_objs, child_objs = event["start_state_objects"], event["end_state_objects"]
        else:
            parent_objs, child_objs = event["end_state_objects"], event["start_state_objects"]

        parent = parent_objs[0]
        child = next(obj for obj in child_objs if obj != parent)
        events.append({"parent": parent, "child": child, "frame": event["frame"], "type": event["type"]})

    return frame_paths, fwd_tracks, rev_tracks, fwd_meta, rev_meta, events, pairs


def get_event_node_obj_ids(node, track_to_pair_idxs):
    """ Final obj_ids an event's node stands for. A tubelet that was matched becomes the pairs
    it belongs to; one that was not keeps its own id, since it is an object on its own.
    """
    prefix, track_id = node.split("_")
    pair_idxs = track_to_pair_idxs.get(prefix, {}).get(track_id)
    return [f"Pair_{pair_idx}" for pair_idx in pair_idxs] if pair_idxs else [node]


def subtract_children_from_parents(all_final_masks, events, track_to_pair_idxs):
    """ Remove each child's pixels from its parent, so an object and the thing it split from
    never both claim the same region on a frame. Edits all_final_masks in place.
    """
    for event in events:
        children = get_event_node_obj_ids(event["child"], track_to_pair_idxs)
        parents = get_event_node_obj_ids(event["parent"], track_to_pair_idxs)
        parents = [parent for parent in parents if parent not in children]

        if not parents or not children:
            continue

        frames = sorted({
            frame_idx
            for child in children
            for frame_idx in all_final_masks.get(child, {})
        })

        for frame_idx in frames:
            # Combine all child masks on this frame
            child_mask = union_masks(all_final_masks.get(child, {}).get(frame_idx) for child in children)
            if child_mask is None:
                continue
            child_bmask = rle_to_bmask(child_mask)

            # Remove child pixels from each parent
            for parent in parents:
                parent_masks = all_final_masks.get(parent, {})
                if frame_idx not in parent_masks:
                    continue

                remaining = rle_to_bmask(parent_masks[frame_idx]) & ~child_bmask
                if remaining.any():
                    parent_masks[frame_idx] = bmask_to_rle(remaining)
                else:
                    del parent_masks[frame_idx]


def get_output_dataset(args):
    """ Output namespace of an ablation run, which keeps its artifacts separate. """
    output_dataset = args.dataset
    if args.skip_fp: output_dataset += '_skip_fp'
    if args.skip_temporal_alignment: output_dataset += '_skip_temporal_alignment'
    return output_dataset


def promote_unmerged_tracks(args, img_shape, metas, trimmed, track_to_pair_idxs, event_unpaired, pair_masks):
    """ Recover the tubelets that matched nothing and that no event names. One is kept 
    over the frames where it stays big enough and mostly unclaimed by the objects already 
    decided.
    """
    h, w = img_shape
    promoted = {"F": {}, "R": {}}
    # Find unmatched tracks, longest first within each direction
    candidates = []
    for prefix in ("F", "R"):
        tracks = [
            (prefix, track_id, info["frames"])
            for track_id, info in metas[prefix].items()
            if track_id not in track_to_pair_idxs[prefix] and track_id not in event_unpaired[prefix]
        ]
        candidates.extend(sorted(tracks, key=lambda x: len(x[2]), reverse=True))
    for prefix, track_id, frames in candidates:
        clean_frames = []
        for frame_idx in frames:
            mask = trimmed[prefix].get(frame_idx, {}).get(track_id)
            if not mask:
                continue
            bmask = rle_to_bmask(mask)
            area = bmask.sum()
            if area == 0 or area / (h * w) <= args.min_area_perc:
                continue
            # Pixels already claimed by other objects
            claimed = np.zeros((h, w), dtype=bool)
            for frame_masks in pair_masks.values():
                if (pair_mask := frame_masks.get(frame_idx)) is not None:
                    claimed |= rle_to_bmask(pair_mask)
            for direction in ("F", "R"):
                frame_tracks = trimmed[direction].get(frame_idx, {})
                claimed_ids = set(event_unpaired[direction])
                claimed_ids.update(
                    promoted_id for promoted_id, promoted_frames in promoted[direction].items()
                    if frame_idx in promoted_frames
                )

                for claimed_id in claimed_ids:
                    if claimed_mask := frame_tracks.get(claimed_id):
                        claimed |= rle_to_bmask(claimed_mask)

            overlap = (claimed & bmask).sum() / area
            if overlap < args.overlap_thresh:
                clean_frames.append(frame_idx)

        if len(clean_frames) >= args.required_clean_frames:
            promoted[prefix][track_id] = set(range(min(clean_frames), max(clean_frames) + 1))

    return promoted

def merge_prompt_seeds(cfg, args, instance, all_final_masks, fwd_tracks, fwd_meta,
                       pairs, event_unpaired_fwd, promoted_fwd):
    """ Fold the tubelets seeded from the first-frame annotation back into the object they
    became, and record that mapping. Does nothing on a dataset tracked without prompts.
    """
    tracks_dir = osp.join(cfg.paths.intermdir, f"tubelets_{args.dataset}_{cfg.tubelet.entity_method}")
    prompt_seed_path = osp.join(tracks_dir, f"{instance}_prompt_seeds.json")
    if not osp.exists(prompt_seed_path):
        return

    with open(prompt_seed_path, "r") as f:
        prompt_seed_map = json.load(f)

    obj_id_by_node = {
        node: f"Pair_{pair_idx}"
        for pair_idx, pair_members in enumerate(pairs)
        for node in pair_members
    }
    prompt_final_map = {}
    for prompt_idx, seed_id in prompt_seed_map.items():
        prompt_idx, seed_id = str(prompt_idx), str(seed_id)
        seed_node = f"F_{seed_id}"
        obj_id = obj_id_by_node.get(seed_node)
        if obj_id is None and (seed_id in event_unpaired_fwd or seed_id in promoted_fwd):
            obj_id = seed_node
        elif obj_id is None and seed_id in fwd_meta:
            obj_id = f"Prompt_{prompt_idx}"
        if obj_id is not None:
            prompt_final_map[prompt_idx] = obj_id

    map_path = osp.join(cfg.paths.outdir, get_output_dataset(args), f"{instance}_prompt_final_map.json")
    os.makedirs(osp.dirname(map_path), exist_ok=True)
    with open(map_path, "w") as f:
        json.dump(prompt_final_map, f)

    for prompt_idx, obj_id in prompt_final_map.items():
        seed_id = str(prompt_seed_map.get(str(prompt_idx)))
        if seed_id == "None":
            continue
        for frame_idx, frame_tracks in fwd_tracks.items():
            if seed_id not in frame_tracks:
                continue
            prompt_mask = frame_tracks[seed_id]
            if frame_idx in all_final_masks[obj_id]:
                combined_bmask = union_pair_masks(all_final_masks[obj_id][frame_idx], prompt_mask)
                if combined_bmask is not None:
                    all_final_masks[obj_id][frame_idx] = bmask_to_rle(combined_bmask)
            else:
                all_final_masks[obj_id][frame_idx] = prompt_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-d", "--dataset", default="ours")
    parser.add_argument("-i", "--instance", required=True)
    parser.add_argument("--overlap_thresh", type=float, default=0.20)
    parser.add_argument("--min_area_perc", type=float, default=0.0016)
    parser.add_argument("--required_clean_frames", type=int, default=3)
    parser.add_argument("--skip_fp", action="store_true")
    parser.add_argument("--skip_temporal_alignment", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    (
        frame_paths,
        fwd_tracks,
        rev_tracks,
        fwd_meta,
        rev_meta,
        events,
        pairs,
    ) = load_final_mask_inputs(cfg, args, args.instance)
    
    start_time = time.time()
    with Image.open(frame_paths[0]) as img:
        img_shape = (img.height, img.width)
    total_frames = len(frame_paths)
    metas = {"F": fwd_meta, "R": rev_meta}
    trimmed = {
        "F": {frame_idx: masks.copy() for frame_idx, masks in fwd_tracks.items()},
        "R": {frame_idx: masks.copy() for frame_idx, masks in rev_tracks.items()},
    }

    # Find which pairs each tubelet belongs to
    track_to_pair_idxs = {"F": defaultdict(list), "R": defaultdict(list)}
    for pair_idx, (fwd_node, rev_node) in enumerate(pairs):
        track_to_pair_idxs["F"][fwd_node.split("_")[1]].append(pair_idx)
        track_to_pair_idxs["R"][rev_node.split("_")[1]].append(pair_idx)

    # Gather tubelets an event names but that were never matched
    event_unpaired = {"F": set(), "R": set()}
    for event in events:
        for node in (event["parent"], event["child"]):
            prefix, track_id = node.split("_")
            if track_id not in track_to_pair_idxs[prefix]:
                event_unpaired[prefix].add(track_id)

    # Rule: a child only exists after it splits off, or before it merges away
    pair_bounds = {pair_idx: [0, total_frames] for pair_idx in range(len(pairs))}
    pair_has_event = {pair_idx: False for pair_idx in range(len(pairs))}
    for event in events:
        prefix, child_id = event["child"].split("_")
        vlm_frame = event["frame"]
        is_composition = event["type"] == "composition"

        for pair_idx in track_to_pair_idxs[prefix].get(child_id, []):
            pair_has_event[pair_idx] = True
            if is_composition:
                pair_bounds[pair_idx][1] = min(pair_bounds[pair_idx][1], vlm_frame)
            else:
                pair_bounds[pair_idx][0] = max(pair_bounds[pair_idx][0], vlm_frame)

        if child_id in event_unpaired[prefix]:
            for frame_time in list(trimmed[prefix].keys()):
                if child_id in trimmed[prefix][frame_time]:
                    if (is_composition and frame_time > vlm_frame) or (not is_composition and frame_time < vlm_frame):
                        del trimmed[prefix][frame_time][child_id]

    # Frames each pair is kept on, following the rule above
    pair_valid_frames = {}
    for pair_idx, (fwd_node, rev_node) in enumerate(pairs):
        member_frames = (fwd_meta[fwd_node.split("_")[1]]["frames"]
                         | rev_meta[rev_node.split("_")[1]]["frames"])

        min_f, max_f = pair_bounds[pair_idx]
        pair_valid_frames[pair_idx] = (
            {f for f in member_frames if min_f <= f <= max_f} if pair_has_event[pair_idx] else member_frames
        )

    # Drop each paired tubelet's masks from the frames its pairs are not kept on 
    for prefix in ("F", "R"):
        valid_frames = defaultdict(set)
        for track_id, pair_idxs in track_to_pair_idxs[prefix].items():
            for pair_idx in pair_idxs:
                valid_frames[track_id].update(pair_valid_frames[pair_idx])

        for frame_time in list(trimmed[prefix].keys()):
            for track_id in list(trimmed[prefix][frame_time].keys()):
                if track_id in valid_frames and frame_time not in valid_frames[track_id]:
                    del trimmed[prefix][frame_time][track_id]

    pair_masks = get_pair_masks(
        pair_valid_frames, pairs, trimmed["F"], trimmed["R"]
    )

    # Add tubelets that no pair and no event accounts for, when they stay big enough and
    # mostly unclaimed by the objects already decided
    promoted = promote_unmerged_tracks(
        args, img_shape, metas, trimmed, track_to_pair_idxs, event_unpaired, pair_masks
    )

    # Build a final, unified spatiotemporal partition 
    all_final_masks = defaultdict(dict)
    for frame_idx in range(total_frames):
        # Add each matched pair as one object
        for pair_idx in range(len(pairs)):
            pair_mask = pair_masks[pair_idx].get(frame_idx)
            if pair_mask is not None:
                all_final_masks[f"Pair_{pair_idx}"][frame_idx] = pair_mask

        # Add tubelets either because an event named it or it was promoted
        for prefix in ("F", "R"):
            frame_tracks = trimmed[prefix].get(frame_idx, {})
            standalone = set(event_unpaired[prefix])
            standalone.update(
                track_id for track_id, valid_frames in promoted[prefix].items()
                if frame_idx in valid_frames
            )
            for track_id in standalone:
                mask = frame_tracks.get(track_id)
                if mask and rle_to_bmask(mask).sum() > 0:
                    all_final_masks[f"{prefix}_{track_id}"][frame_idx] = mask

    merge_prompt_seeds(
        cfg, args, args.instance, all_final_masks, fwd_tracks, fwd_meta,
        pairs, event_unpaired["F"], promoted["F"],
    )
    # A child and the thing it split from must not both claim the same pixels
    subtract_children_from_parents(all_final_masks, events, track_to_pair_idxs)
    save_timing(cfg, f'build_masks_{args.dataset}', args.instance, time.time() - start_time, len(frame_paths))

    out_path = osp.join(cfg.paths.outdir, get_output_dataset(args), f"{args.instance}_final_masks.json")
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_final_masks, f)
    print(f"Final masks saved to {out_path}")


if __name__ == "__main__":
    main()
