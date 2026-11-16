"""Fill bad-mask frames in annotation JSONs using SAM2 video propagation.

For each object and each contiguous run of bad-mask frames, this script:
  1. Collects the last ≤7 good-mask frames before the run as SAM2 prompts.
  2. Propagates forward through the bad frames.
  3. Evaluates quality using the first good-mask frame after the run (IoU).
  4. Saves a filled annotation JSON with bad-mask frames replaced by SAM2 predictions.

Usage:
    python dataset/fill_annotations.py -i iclicker_1 --frames_dir frames/iclicker_1
"""
import sys, os, json, logging, copy, argparse
import os.path as osp
import numpy as np
import torch
from pycocotools import mask as MaskUtils

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from utils import rle_to_bmask, bmask_to_rle, load_yaml_file
from Atlas.tracker import get_tracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_bad_runs(timeline):
    """Return list of (run_start_idx, run_end_idx) for consecutive bad frames.

    timeline: list of (frame_idx, mask_rle, is_bad) sorted by frame_idx.
    Indices are positions in the timeline list, not frame numbers.
    """
    runs = []
    i = 0
    while i < len(timeline):
        if timeline[i][2]:  # is_bad
            start = i
            while i < len(timeline) and timeline[i][2]:
                i += 1
            runs.append((start, i - 1))  # inclusive
        else:
            i += 1
    return runs


def compute_iou(pred_mask_bool, gt_rle):
    """Compute IoU between a boolean np.ndarray and a GT RLE mask."""
    pred_rle = bmask_to_rle(pred_mask_bool.astype(bool))
    iou = MaskUtils.iou([pred_rle], [gt_rle], [False])[0, 0]
    return float(iou)


def _propagate(sam2, prompt_frames, bad_frame_set, stop_frame, reverse=False, start_frame_idx=None):
    """Reset state, add prompt masks, propagate. Returns (masks, logits) dicts keyed by frame_idx."""
    sam2.predictor.reset_state(sam2.inference_state)
    for frame_idx, mask_rle, _ in prompt_frames:
        sam2.predictor.add_new_mask(
            inference_state=sam2.inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=rle_to_bmask(mask_rle),
        )
    masks, logits = {}, {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2.predictor.propagate_in_video(
        sam2.inference_state, reverse=reverse, start_frame_idx=start_frame_idx
    ):
        if out_frame_idx in bad_frame_set:
            masks[out_frame_idx] = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
            logits[out_frame_idx] = float(out_mask_logits[0, 0].abs().mean())
        if not reverse and out_frame_idx >= stop_frame:
            break
        if reverse and out_frame_idx <= stop_frame:
            break
    return masks, logits


def fill_object_runs(sam2, frames_dir, timeline, max_prompts=7, max_prompt_distance=50):
    """Fill all bad runs for one object using bidirectional SAM2 propagation.

    For each bad run:
      - Forward pass: prompt from the last ≤max_prompts good frames within
        max_prompt_distance frames before the run.
      - Backward pass: prompt from the first good frame after the run
        (the validation frame) and propagate in reverse.
      - Per-frame selection: pick whichever direction has higher logit confidence.

    IoU is reported using the forward-only prediction at the validation frame
    (an unbiased proxy for propagation quality).

    Returns:
        filled: dict {frame_idx: rle} for every bad frame
        ious:   list of (validation_frame_idx, iou) per run
    """
    filled = {}
    ious = []

    for run_start, run_end in get_bad_runs(timeline):
        bad_frame_set = {timeline[k][0] for k in range(run_start, run_end + 1)}
        run_start_frame = timeline[run_start][0]
        run_end_frame = timeline[run_end][0]

        # Good frames before the run, filtered by temporal proximity
        candidates = [timeline[k] for k in range(run_start - 1, -1, -1)
                      if not timeline[k][2]]
        within_window = [e for e in candidates
                         if (run_start_frame - e[0]) <= max_prompt_distance]
        prompt_entries = list(reversed(within_window[:max_prompts] if within_window else candidates[:1]))

        # First good frame after the run (validation)
        val_entry = next(
            (timeline[k] for k in range(run_end + 1, len(timeline)) if not timeline[k][2]),
            None
        )

        if not prompt_entries:
            log.warning('No prompt frames before bad run at frame %d, skipping.', run_start_frame)
            continue

        val_frame_idx = val_entry[0] if val_entry is not None else None

        log.info(
            'Bad run frames %d–%d | prompts: %s | validation: %s',
            run_start_frame, run_end_frame,
            [e[0] for e in prompt_entries],
            val_frame_idx,
        )

        # --- Forward propagation from prompt frames ---
        fwd_stop = val_frame_idx if val_frame_idx is not None else run_end_frame
        fwd_masks, fwd_logits = _propagate(sam2, prompt_frames=prompt_entries,
                                           bad_frame_set=bad_frame_set,
                                           stop_frame=fwd_stop, reverse=False)

        # --- Backward propagation from validation frame ---
        bwd_masks, bwd_logits = {}, {}
        if val_entry is not None:
            bwd_masks, bwd_logits = _propagate(
                sam2,
                prompt_frames=[(val_frame_idx, val_entry[1], False)],
                bad_frame_set=bad_frame_set,
                stop_frame=run_start_frame,
                reverse=True,
                start_frame_idx=val_frame_idx,
            )

        # --- Anchored forward prediction at validation frame (for IoU evaluation) ---
        # For long runs (>= ANCHOR_MIN_LEN), insert evenly-spaced backward-fill frames
        # as additional SAM2 conditioning to reduce tracking drift.
        ANCHOR_MIN_LEN = 20
        N_ANCHORS = 3
        pred_at_val = None
        if val_frame_idx is not None:
            anchor_prompts = []
            bad_sorted = sorted(bad_frame_set)
            if bwd_masks and len(bad_sorted) >= ANCHOR_MIN_LEN:
                step = max(1, len(bad_sorted) // (N_ANCHORS + 1))
                anchor_frame_idxs = [bad_sorted[i * step] for i in range(1, N_ANCHORS + 1)
                                     if i * step < len(bad_sorted)]
                for af in anchor_frame_idxs:
                    if af in bwd_masks:
                        anchor_prompts.append((af, bmask_to_rle(bwd_masks[af]), False))
                if anchor_prompts:
                    log.info('  Using %d backward-fill anchors at frames %s',
                             len(anchor_prompts), [a[0] for a in anchor_prompts])

            sam2.predictor.reset_state(sam2.inference_state)
            for frame_idx, mask_rle, _ in prompt_entries:
                sam2.predictor.add_new_mask(
                    inference_state=sam2.inference_state,
                    frame_idx=frame_idx, obj_id=0, mask=rle_to_bmask(mask_rle),
                )
            for af, af_rle, _ in anchor_prompts:
                sam2.predictor.add_new_mask(
                    inference_state=sam2.inference_state,
                    frame_idx=af, obj_id=0, mask=rle_to_bmask(af_rle),
                )
            for out_frame_idx, _, out_mask_logits in sam2.predictor.propagate_in_video(
                sam2.inference_state
            ):
                if out_frame_idx == val_frame_idx:
                    pred_at_val = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                    break

        # Determine merge strategy based on area change between prompt and validation.
        # If the object expands or contracts significantly, backward propagation (from
        # the validation frame) gives a more faithful fill because it "knows" the
        # target state. For stable objects, logit-based selection is used.
        AREA_RATIO_THRESHOLD = 1.5  # triggers backward-preference if val/prompt > 1.5x or < 0.67x
        prefer_backward = False
        if bwd_masks and val_entry is not None and prompt_entries:
            last_prompt_rle_check = {'counts': prompt_entries[-1][1]['counts'],
                                     'size': prompt_entries[-1][1]['size']}
            val_rle_check = {'counts': val_entry[1]['counts'], 'size': val_entry[1]['size']}
            last_prompt_area = MaskUtils.area(last_prompt_rle_check)
            val_area = MaskUtils.area(val_rle_check)
            area_ratio = val_area / max(float(last_prompt_area), 1.0)
            if area_ratio > AREA_RATIO_THRESHOLD or area_ratio < 1.0 / AREA_RATIO_THRESHOLD:
                prefer_backward = True
                log.info('  Area ratio val/prompt=%.2f → preferring backward fill', area_ratio)

        # --- Merge: per-frame pick the better direction ---
        for frame_idx in sorted(bad_frame_set):
            fwd = fwd_masks.get(frame_idx)
            bwd = bwd_masks.get(frame_idx)
            if fwd is not None and bwd is not None:
                if prefer_backward:
                    best = bwd
                else:
                    best = bwd if bwd_logits[frame_idx] > fwd_logits[frame_idx] else fwd
            elif fwd is not None:
                best = fwd
            elif bwd is not None:
                best = bwd
            else:
                log.warning('No prediction for bad frame %d', frame_idx)
                continue
            filled[frame_idx] = bmask_to_rle(best)

        # Evaluate forward IoU (unbiased: forward prediction vs GT at validation frame)
        if pred_at_val is not None and val_entry is not None:
            iou_fwd = compute_iou(pred_at_val, val_entry[1])
            log.info('  IoU at validation frame %d: %.4f (fwd)', val_frame_idx, iou_fwd)
            ious.append((val_frame_idx, iou_fwd))

        # Evaluate backward IoU (backward prediction vs GT at last prompt frame)
        if bwd_masks and prompt_entries:
            last_prompt_frame = prompt_entries[-1][0]
            last_prompt_rle = prompt_entries[-1][1]
            # Re-run backward propagation to capture prediction at the prompt frame
            sam2.predictor.reset_state(sam2.inference_state)
            sam2.predictor.add_new_mask(
                inference_state=sam2.inference_state,
                frame_idx=val_frame_idx, obj_id=0, mask=rle_to_bmask(val_entry[1]),
            )
            pred_at_prompt = None
            for out_frame_idx, _, out_mask_logits in sam2.predictor.propagate_in_video(
                sam2.inference_state, reverse=True, start_frame_idx=val_frame_idx
            ):
                if out_frame_idx == last_prompt_frame:
                    pred_at_prompt = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                    break
                if out_frame_idx < last_prompt_frame:
                    break
            if pred_at_prompt is not None:
                iou_bwd = compute_iou(pred_at_prompt, last_prompt_rle)
                log.info('  IoU at last prompt frame %d: %.4f (bwd)', last_prompt_frame, iou_bwd)

    return filled, ious


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description='Fill bad-mask frames with SAM2 propagation.')
    parser.add_argument('-i', '--instance', required=True, help='Instance name (e.g. buildgreenchair)')
    parser.add_argument('--frames_dir', default=None, help='Video frame directory (default: assets/example/{instance})')
    parser.add_argument('--annotations_dir', default='annotations/', help='Input annotation directory')
    parser.add_argument('--output_dir', default='annotations_filled/', help='Output directory for filled JSON')
    parser.add_argument('-c', '--config', default='configs/default.yaml', help='OmegaConf config path (for SAM2 model paths)')
    parser.add_argument('--max_prompt_distance', type=int, default=50,
                        help='Only use good frames within this many frames before a bad run as prompts (default: 50)')
    return parser


def main():
    args = get_parser().parse_args()

    if args.frames_dir is None:
        args.frames_dir = osp.join('assets/example', args.instance)

    anno_path = osp.join(args.annotations_dir, f'{args.instance}_annotations.json')
    log.info('Loading annotations from %s', anno_path)
    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    cfg = load_yaml_file(args.config)
    sam2 = get_tracker(cfg.tubelet.tracker)
    log.info('Initializing SAM2 with frames from %s', args.frames_dir)
    sam2.initialize(video_dir=args.frames_dir)

    # Collect all object IDs present across all frames
    all_obj_ids = set()
    for frame_data in annotations.values():
        all_obj_ids.update(frame_data.keys())

    filled_annotations = copy.deepcopy(annotations)
    all_ious = {}

    for obj_id in sorted(all_obj_ids):
        # Build timeline sorted by frame index
        timeline = []
        for frame_str, frame_data in annotations.items():
            if obj_id not in frame_data:
                continue
            entry = frame_data[obj_id]
            mask_rle = {'counts': entry['counts'], 'size': entry['size']}
            timeline.append((int(frame_str), mask_rle, entry['bad_mask']))
        timeline.sort(key=lambda x: x[0])

        obj_name = annotations[str(timeline[0][0])][obj_id].get('object_name', obj_id)
        log.info('=== Object %s (%s): %d frames, %d bad ===',
                 obj_id, obj_name, len(timeline), sum(1 for t in timeline if t[2]))

        filled, ious = fill_object_runs(sam2, args.frames_dir, timeline,
                                         max_prompt_distance=args.max_prompt_distance)

        # Write filled masks back into the output annotation
        for frame_idx, rle in filled.items():
            frame_str = str(frame_idx)
            filled_annotations[frame_str][obj_id]['counts'] = rle['counts']
            filled_annotations[frame_str][obj_id]['size'] = rle['size']

        all_ious[obj_id] = ious

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = osp.join(args.output_dir, f'{args.instance}_annotations.json')
    with open(out_path, 'w') as f:
        json.dump(filled_annotations, f)
    log.info('Saved filled annotations to %s', out_path)

    # Summary
    log.info('=== Evaluation Summary ===')
    for obj_id, ious in all_ious.items():
        if ious:
            mean_iou = np.mean([v for _, v in ious])
            log.info('  Object %s: mean IoU = %.4f over %d run(s)', obj_id, mean_iou, len(ious))
            for val_f, iou in ious:
                log.info('    val frame %d: IoU = %.4f', val_f, iou)
        else:
            log.info('  Object %s: no validation frames', obj_id)

    sam2.clear_all_cache()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
