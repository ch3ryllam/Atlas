"""
Stage 2 of the VLM + detector baseline: track each object the VLM named through the whole
video in both directions and merge tracks that meet across an event, so every object ends up
with one mask track instead of a per-event detection.

Usage: python3 baselines/vlm_detector/complete_tracks.py -c configs/default.yaml -d ours -i crackers --tracker sam3
"""

import os, json, glob, argparse, sys, logging
import os.path as osp

from sam2.build_sam import build_sam2_video_predictor

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, rle_to_bmask, bmask_to_rle
from baselines.backends.tracker import SAM3VideoPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def _mask_iou(m1, m2):
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0

def _extract_seeds(bl_masks, events):
    seed_info = {}
    for bl_id, frame_dict in bl_masks.items():
        
        event = None
        for ev in events:
            if bl_id in ev["start_state_objects"] or bl_id in ev["end_state_objects"]:
                event = ev
                break
        assert event is not None, f"{bl_id} not found in any event"

        info = []
        before_frame = min(int(f) for f in event["det_parent_loc"].keys())
        for seed_frame_str in frame_dict.keys():
            rle = frame_dict[seed_frame_str]
            if not rle:
                log.warning(f"{bl_id}: empty RLE at seed frame {seed_frame_str}, skipping")
                continue
            bmask = rle_to_bmask(rle)
            seed_frame_idx = int(seed_frame_str)
            # Split on the event's before-detection frame, as before_frames does, so that
            # an event at frame 0 still has a before seed.
            if seed_frame_idx <= before_frame:
                frame_type = "before"
            else:
                frame_type = "after"
            
            info.append({"seed_frame": int(seed_frame_str), "mask": bmask, "frame_type": frame_type})
            
        seed_info[bl_id] = info 
    return seed_info

def find_best_overlap(track_mask, ev_f, seed_info, matched_before_bl_ids):
    # Check for IoU with before objects at this event
    before_bl_ids = [bl_id for bl_id in ev_f["start_state_objects"] if bl_id not in matched_before_bl_ids]  # only consider unmatched before objects
    iou_list = []
    for before_bl_id in before_bl_ids:
        before_bl_info = [x for x in seed_info[before_bl_id] if x["frame_type"] == "before"]
        assert len(before_bl_info) == 1, f"Expected exactly one before seed for {before_bl_id}, got {len(before_bl_info)}"
        before_bl_info = before_bl_info[0]
        iou = _mask_iou(track_mask, before_bl_info["mask"])
        iou_list.append((before_bl_id, iou))
    if iou_list:
        best_before_bl_id, best_iou = max(iou_list, key=lambda x: x[1])
    else:
        best_before_bl_id, best_iou = None, None
    return best_before_bl_id, best_iou


def connect_all_objects(predictor, inference_state, seed_info, events, iou_thresh):
    """
    Track all BL_* objects bidirectionally and connect tracks across events based on IoU.
    For all objects in before event, track into the past until either it matches an after object or it reaches the first frame.
    For all objects in after event, track into the future until either it matches a before object or it reaches the last frame.
    Finally, track all before or after objects to the event frame.

    Since we are matching objects across events, we just need to keep track of the final objects after merging.
    """
    # Map each BL_* id to its associated event (first occurrence, matching _extract_seeds logic)
    bl_to_event = {}
    for ev in events:
        for bl_id in ev["start_state_objects"] + ev["end_state_objects"]:
            if bl_id not in bl_to_event:
                bl_to_event[bl_id] = ev
    
    frame_to_event = {ev["frame"]: ev for ev in events}
    ckpt_frames = sorted(frame_to_event.keys())
    before_frames = {min(int(f) for f in ev['det_parent_loc'].keys()): ev for ev in events}
    
    final_masks = {}
    merge_pairs = []  # (before_bl_id, after_bl_id)

    ## Forward tracking: tracking each after objects into the future, merge with any before objects if IoU > thresh
    matched_before_bl_ids = set()
    for ckpt_f in ckpt_frames:
        ev = frame_to_event[ckpt_f]
        log.info(f"Processing event at frame {ckpt_f}")

        for bl_id in ev["end_state_objects"]:
            bl_track = {}
            bl_info = [x for x in seed_info[bl_id] if x["frame_type"] == "after"]
            assert len(bl_info) == 1, f"Expected exactly one after seed for {bl_id}, got {len(bl_info)}"
            bl_info = bl_info[0]

            log.info(f"  Tracking after object {bl_id} forward")
            predictor.reset_state(inference_state)
            predictor.add_new_mask(inference_state, frame_idx=bl_info["seed_frame"], obj_id=0, mask=bl_info["mask"])

            for f, _, logits in predictor.propagate_in_video(inference_state):
                pred_mask = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                bl_track[str(f)] = bmask_to_rle(pred_mask)
                if f in before_frames:
                    future_event = before_frames[f]
                    before_bl_id, iou = find_best_overlap(pred_mask, future_event, seed_info, matched_before_bl_ids)
                    if iou is not None and iou > iou_thresh:
                        log.info(f"    → Merging with before object {before_bl_id} at event frame {f} (IoU={iou:.2f})")
                        merge_pairs.append((bl_id, before_bl_id, iou))
                        matched_before_bl_ids.add(before_bl_id)
                        break
                
            log.info(f"    → Track ends at frame {f}")
            final_masks[bl_id] = bl_track
    
    ## Backward tracking: tracking each before object into the past, merge with any after objects if IoU > thresh
    matched_after_bl_ids = set()
    for ckpt_f in reversed(ckpt_frames):
        ev = frame_to_event[ckpt_f]
        log.info(f"Processing event at frame {ckpt_f}")

        for bl_id in ev["start_state_objects"]:
            if bl_id in matched_before_bl_ids:
                log.info(f"  Skipping before object {bl_id} since it's already matched with an after object")
                continue

            bl_track = {}
            bl_info = [x for x in seed_info[bl_id] if x["frame_type"] == "before"]
            assert len(bl_info) == 1, f"Expected exactly one before seed for {bl_id}, got {len(bl_info)}"
            bl_info = bl_info[0]

            log.info(f"  Tracking before object {bl_id} backward")
            predictor.reset_state(inference_state)
            predictor.add_new_mask(inference_state, frame_idx=bl_info["seed_frame"], obj_id=0, mask=bl_info["mask"])

            for f, _, logits in predictor.propagate_in_video(inference_state, reverse=True):
                pred_mask = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                bl_track[str(f)] = bmask_to_rle(pred_mask)

            log.info(f"    → Track ends at frame {f}")
            if bl_id in final_masks:
                final_masks[bl_id].update(bl_track)
            else:
                final_masks[bl_id] = bl_track
    
    ## Fill-in gaps for before and after objects
    for bl_id, bl_info in seed_info.items():
        if bl_id not in final_masks:
            final_masks[bl_id] = {}
        if len(bl_info) == 2:
            bl_start_info = [x for x in bl_info if x["frame_type"] == "before"][0]
            bl_end_info = [x for x in bl_info if x["frame_type"] == "after"][0]

            predictor.reset_state(inference_state)
            predictor.add_new_mask(inference_state, frame_idx=bl_start_info["seed_frame"], obj_id=0, mask=bl_start_info["mask"])
            for f, _, logits in predictor.propagate_in_video(inference_state):
                pred_mask = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                if f > bl_end_info["seed_frame"]:
                    break
                if str(f) not in final_masks[bl_id]:
                    final_masks[bl_id][str(f)] = bmask_to_rle(pred_mask)
        elif len(bl_info) == 1 and bl_info[0]["frame_type"] == "before":
            bl_start_info = bl_info[0]
            bl_env = bl_to_event[bl_id]

            predictor.reset_state(inference_state)
            predictor.add_new_mask(inference_state, frame_idx=bl_start_info["seed_frame"], obj_id=0, mask=bl_start_info["mask"])
            for f, _, logits in predictor.propagate_in_video(inference_state):
                pred_mask = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                if f > bl_env["frame"]:
                    break
                if str(f) not in final_masks[bl_id]:
                    final_masks[bl_id][str(f)] = bmask_to_rle(pred_mask)
        elif len(bl_info) == 1 and bl_info[0]["frame_type"] == "after":
            bl_end_info = bl_info[0]
            bl_env = bl_to_event[bl_id]

            predictor.reset_state(inference_state)
            predictor.add_new_mask(inference_state, frame_idx=bl_end_info["seed_frame"], obj_id=0, mask=bl_end_info["mask"])
            for f, _, logits in predictor.propagate_in_video(inference_state, reverse=True):
                pred_mask = (logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                if f < bl_env["frame"]:
                    break
                if str(f) not in final_masks[bl_id]:
                    final_masks[bl_id][str(f)] = bmask_to_rle(pred_mask)
        else:
            raise ValueError(f"Unexpected number of seeds for {bl_id}: {len(bl_info)}")

    ## Create mappings
    pair_id_2_bl_ids = {}
    bl_id_2_pair_id = {}
    for bl_id_a, bl_id_b, iou in merge_pairs:
        pair_id = f"Pair_{len(pair_id_2_bl_ids)}"
        pair_id_2_bl_ids[pair_id] = (bl_id_a, bl_id_b)
        bl_id_2_pair_id[bl_id_a] = pair_id
        bl_id_2_pair_id[bl_id_b] = pair_id

    # Update event names
    for ev in events:
        org_start_list = ev['start_state_objects']
        ev['start_state_objects'] = [bl_id_2_pair_id.get(x, x) for x in org_start_list]
        org_end_list = ev['end_state_objects']
        ev['end_state_objects'] = [bl_id_2_pair_id.get(x, x) for x in org_end_list]

    # Update masks
    for pair_id, (bl_id_a, bl_id_b) in pair_id_2_bl_ids.items():
        final_masks[pair_id] = final_masks.get(bl_id_a, {})
        final_masks[pair_id].update(final_masks.get(bl_id_b, {}))
        if bl_id_a in final_masks:
            final_masks.pop(bl_id_a)
        if bl_id_b in final_masks:
            final_masks.pop(bl_id_b)

    return final_masks, events

# Example: python3 baselines/complete_obj_tracks.py -c=configs/default.yaml -d ours -i crackers
def main():
    p = argparse.ArgumentParser(description="Track BL_* objects through full video and merge into Pair_* state graph")
    p.add_argument('-c', '--config',         required=True)
    p.add_argument('-d', '--dataset',        required=True)
    p.add_argument('-i', '--instance',       required=True)
    p.add_argument('-p', '--pred',           default=None, help="default: <cfg.vlm.model_name>_gdino_16")
    p.add_argument('--iou_thresh',           type=float, default=0.5)
    p.add_argument('--tracker',              default="sam2", help="sam2 | sam3")
    p.add_argument('--sam2_model_cfg',       default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument('--sam2_weights',         default="_ckpts/sam2.1_hiera_large.pt")
    args = p.parse_args()

    cfg      = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    instance = args.instance
    out_dir  = cfg.paths.outdir + '_baselines'
    device   = f"cuda:0"
    pred     = args.pred or f"{cfg.vlm.model_name}_gdino_16"

    image_dir   = osp.join(data_cfg.image_dir, instance)
    frame_paths = sorted(glob.glob(osp.join(image_dir, data_cfg.image_format)))
    if not frame_paths:
        raise FileNotFoundError(f"No frames at {osp.join(image_dir, data_cfg.image_format)}")
    log.info(f"Found {len(frame_paths)} frames for '{instance}'")

    # ── Step 1: Load inputs ───────────────────────────────────────────────────
    events_path = osp.join(out_dir, pred, f"{instance}_events.json")
    masks_path  = osp.join(out_dir, pred, f"{instance}_final_masks.json")
    events   = json.load(open(events_path))
    bl_masks = json.load(open(masks_path))

    if not events:
        # Still write empty final outputs, so the video is evaluated (and not rerun) like the others
        log.info("No events found. Writing empty outputs.")
        out_subdir = osp.join(out_dir, pred + '_final')
        os.makedirs(out_subdir, exist_ok=True)
        json.dump([], open(osp.join(out_subdir, f"{instance}_events.json"), 'w'))
        json.dump([], open(osp.join(out_subdir, f"{instance}_pairs.json"), 'w'))
        json.dump({}, open(osp.join(out_subdir, f"{instance}_final_masks.json"), 'w'))
        return

    all_bl_ids = set()
    for ev in events:
        all_bl_ids.update(ev["start_state_objects"])
        all_bl_ids.update(ev["end_state_objects"])
    log.info(f"Unique BL_* objects: {sorted(all_bl_ids)}")

    # ── Step 2: Extract seeds ─────────────────────────────────────────────────
    seed_info = _extract_seeds(bl_masks, events)
    log.info(f"Seeds extracted for: {sorted(seed_info.keys())}")

    # ── Step 3: Initialize the video tracker once ───────────────────────────
    log.info(f"Initializing {args.tracker} video predictor...")
    if args.tracker == "sam3":
        predictor       = SAM3VideoPredictor(device=device)
        inference_state = predictor.init_state(frame_paths)
    else:
        predictor       = build_sam2_video_predictor(args.sam2_model_cfg, args.sam2_weights, device=device)
        inference_state = predictor.init_state(video_path=image_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
    log.info(f"{args.tracker} inference state initialized")

    # ── Step 4: Track each object bidirectionally ─────────────────────────────
    final_masks, updated_events = connect_all_objects(predictor, inference_state, seed_info, events, args.iou_thresh)

    # ── Save ──────────────────────────────────────────────────────────
    out_subdir = osp.join(out_dir, pred + '_final')
    os.makedirs(out_subdir, exist_ok=True)
    events_path_final = osp.join(out_subdir, f"{instance}_events.json")
    pairs_path_final = osp.join(out_subdir, f"{instance}_pairs.json")
    masks_path_final = osp.join(out_subdir, f"{instance}_final_masks.json")
    json.dump(updated_events, open(events_path_final, 'w'), indent=2)
    json.dump(final_masks,    open(masks_path_final,  'w'))
    json.dump([], open(pairs_path_final, 'w'))
    log.info(f"Wrote {len(updated_events)} event(s) → {events_path_final}")
    log.info(f"Wrote masks for {len(final_masks)} Pair_* object(s) → {masks_path_final}")

if __name__ == '__main__':
    main()
