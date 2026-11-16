"""
Stage 1 of the VLM + detector baseline: the VLM reads the video (sampled frames, or natively
for Gemini) and reports the transformations it sees, then a detector grounds the object names
it used into masks on the frames around each event.

Usage: python3 baselines/vlm_detector/detect.py -c configs/default.yaml -d ours -i crackers --detector sam3
"""

import os, json, glob, argparse, sys, logging
import os.path as osp
import numpy as np
import imageio.v3 as imageio
from PIL import Image, ImageDraw, ImageFont
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, bmask_to_rle, rle_to_bmask
from baselines.backends import build_vlm, build_detector
from baselines.backends.vlm import parse_timestamp

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _encode_video(frame_paths, fps, out_path):
    """Encode frames to an H.264 mp4 for native video input."""
    import imageio.v2 as iio2
    writer = iio2.get_writer(out_path, fps=fps, codec="libx264", quality=7)
    for p in frame_paths:
        writer.append_data(imageio.imread(p))
    writer.close()


def _get_sam2_mask(predictor, box):
    """Run SAM2 box prompt on a pre-set image. Returns binary mask (H, W)."""
    masks, _, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=np.array(box)[None, :],
        multimask_output=False,
    )
    return bmask_to_rle(masks[0].astype(bool))

def _process_vlm_box(box_str, img_width, img_height):
    """Convert VLM box format (x1, y1, x2, y2) in [0,1] to (x1, y1, x2, y2) in pixel coordinates."""
    print(f"Processing VLM box: {box_str}")
    try:
        box = list(box_str) if box_str is not None else [0, 0, 1, 1]
    except Exception as e:
        log.warning(f"Box Error: Invalid box string '{box_str}': {e}; using full image")
        box = [0, 0, 1, 1]
    try:
        x1, y1, x2, y2 = box
    except Exception as e:  
        log.warning(f"Box Error: Coordinates out of range or invalid: {box}; using full image")
        x1, y1, x2, y2 = 0, 0, 1, 1

    x1_pix = max(0, int(x1 * img_width))
    y1_pix = max(0, int(y1 * img_height))
    x2_pix = min(int(x2 * img_width), img_width)
    y2_pix = min(int(y2 * img_height), img_height)
    return [x1_pix, y1_pix, x2_pix, y2_pix]


# ─── Visualization ────────────────────────────────────────────────────────────

_BOX_COLORS = [(255, 80, 80), (80, 220, 80), (80, 180, 255), (255, 180, 0)]


def _load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_boxes_and_masks(img_np, dets):
    """dets: list of (label, score, box, mask_np_or_None)"""
    img = Image.fromarray(img_np.astype(np.uint8)).convert("RGBA")
    for j, (label, score, box, mask) in enumerate(dets):
        if mask is not None:
            color = _BOX_COLORS[j % len(_BOX_COLORS)]
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            overlay[mask] = (*color, 90)
            img = Image.alpha_composite(img, Image.fromarray(overlay, "RGBA"))
    img  = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(18)
    for j, (label, score, box, mask) in enumerate(dets):
        color = _BOX_COLORS[j % len(_BOX_COLORS)]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        for t in range(3):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        text = f"{label}"
        bb   = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), text, fill=(0, 0, 0), font=font)
    if not dets:
        draw.text((8, 8), "no detections", fill=(255, 80, 80), font=_load_font(20))
    return img


def _add_banner(pil_img, text, color):
    font   = _load_font(20)
    banner = Image.new("RGB", (pil_img.width, 32), (0, 0, 0))
    ImageDraw.Draw(banner).text((6, 6), text, fill=color, font=font)
    out = Image.new("RGB", (pil_img.width, pil_img.height + 32))
    out.paste(banner, (0, 0))
    out.paste(pil_img, (0, 32))
    return out


def visualize_events(events, all_masks, frame_paths, vis_dir):
    os.makedirs(vis_dir, exist_ok=True)
    for i, ev in enumerate(events):
        queried_frame_ind = sorted([int(x) for x in ev["det_parent_loc"].keys()])

        vis_list = []
        for frame_idx in queried_frame_ind:
            frame_idx = str(frame_idx)
            parent_mask = rle_to_bmask(all_masks[f"BL_{i}_parent"][frame_idx])
            dect_results = [
                (ev.get("vlm_parent_name", "parent"), 1.0, ev["det_parent_loc"][frame_idx], parent_mask), 
            ]

            if frame_idx in ev["det_child_loc"]:
                child_mask  = rle_to_bmask(all_masks[f"BL_{i}_child"][frame_idx])
                dect_results.append(
                    (ev.get("vlm_child_name", "child"), 1.0, ev["det_child_loc"][frame_idx], child_mask)
                )
            vis = _draw_boxes_and_masks(imageio.imread(frame_paths[int(frame_idx)]), dect_results)
            vis = _add_banner(vis, f"Frame {frame_idx}  {ev['type']}: P:{ev.get('vlm_parent_name','')} → C:{ev.get('vlm_child_name','')}", (255, 220, 0))
            vis_list.append(vis)
        
        vis_final = Image.new("RGB", (sum(v.width for v in vis_list), vis_list[0].height))
        x_offset = 0
        for vis in vis_list:
            vis_final.paste(vis, (x_offset, 0))
            x_offset += vis.width
        
        out_path = osp.join(vis_dir, f"event_{i:02d}_frame{ev['frame']}_{ev['type']}.jpg")
        vis_final.save(out_path, quality=92)
        log.info(f"Saved visualization → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
# Example: python3 baselines/vlm_det_baseline.py -c configs/default.yaml -d ours -i crackers --vis
def main():
    p = argparse.ArgumentParser(description="VLM baseline")
    p.add_argument('-c', '--config',   required=True)
    p.add_argument('-d', '--dataset',  required=True)
    p.add_argument('-i', '--instance', required=True)
    p.add_argument('--detector',       default="gdino", help="Detector backend: gdino | sam3")
    p.add_argument('--num_vlm_frames', type=int,   default=16)
    p.add_argument('--video_native',   action='store_true', help="Send the full video natively (Gemini only) instead of sampled frames")
    p.add_argument('--video_fps',      type=float, default=5, help="Sampling fps the API applies to native video input")
    p.add_argument('--context_frames', type=int, default=8, help="Frames around event for GDino detection")
    p.add_argument('--gdino_model',    default="IDEA-Research/grounding-dino-base")
    p.add_argument('--box_threshold',  type=float, default=0.0)
    p.add_argument('--text_threshold', type=float, default=0.25)
    p.add_argument('--temp',           type=float, default=0.0)
    p.add_argument('--vis',            action='store_true')
    p.add_argument('--vlm_cache',      action='store_true')
    args = p.parse_args()

    cfg      = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    instance = args.instance
    vlm_name = cfg.vlm.model_name
    baseline_name = f"{vlm_name}_{args.detector}_{'video' if args.video_native else args.num_vlm_frames}"
    out_dir  = osp.join(cfg.paths.outdir + '_baselines', baseline_name)

    os.makedirs(out_dir, exist_ok=True)
    device   = f"cuda:0"

    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, instance, data_cfg.image_format)))
    if not frame_paths:
        raise FileNotFoundError(f"No frames: {osp.join(data_cfg.image_dir, instance, data_cfg.image_format)}")
    total_frames = len(frame_paths)
    log.info(f"Found {total_frames} frames for '{instance}'")

    n_samples      = min(args.num_vlm_frames, len(frame_paths))
    sample_indices = np.linspace(0, len(frame_paths) - 1, n_samples, dtype=int)
    if args.video_native:
        sample_indices = np.arange(total_frames)  # timestamps index the full video directly
    sampled_frames = [imageio.imread(frame_paths[0])] if args.video_native else [imageio.imread(frame_paths[i]) for i in sample_indices]
    H, W = sampled_frames[0].shape[:2]

    # ── VLM stage ────────────────────────────────────────────────────────────
    vlm_cache_path = osp.join(out_dir, f"{instance}_vlm_cache.json")
    if osp.exists(vlm_cache_path) and args.vlm_cache:
        log.info(f"Loading cached VLM output from {vlm_cache_path}")
        cache          = json.load(open(vlm_cache_path))
        raw_events     = cache["raw_events"]
        sample_indices = np.array(cache["sample_indices"])
        log.info(f"Loaded {len(raw_events)} cached event(s)")
    else:
        vlm_backend = build_vlm(vlm_name, temp=args.temp)
        log.info("Querying VLM for transformation events...")
        if args.video_native:
            video_path = osp.join(out_dir, f"{instance}.mp4")
            if not osp.exists(video_path):
                log.info(f"Encoding {total_frames} frames at {data_cfg.fps} fps → {video_path}")
                _encode_video(frame_paths, data_cfg.fps, video_path)
            raw_events = vlm_backend.query_events_video(video_path, fps=args.video_fps)
            for ev in raw_events:
                if "timestamp" in ev:
                    ev["frame_index"] = min(total_frames - 1, max(0, round(parse_timestamp(ev["timestamp"]) * data_cfg.fps)))
        else:
            raw_events = vlm_backend.query_events(sampled_frames)
        log.info(f"VLM detected {len(raw_events)} event(s)")
        json.dump({"raw_events": raw_events, "sample_indices": sample_indices.tolist()},
                  open(vlm_cache_path, 'w'), indent=2)
        log.info(f"Saved VLM cache → {vlm_cache_path}")

    if not raw_events:
        log.info("No events detected. Writing empty outputs.")
        json.dump([], open(osp.join(out_dir, f"{instance}_events.json"), 'w'))
        json.dump([], open(osp.join(out_dir, f"{instance}_pairs.json"), 'w'))
        json.dump({}, open(osp.join(out_dir, f"{instance}_final_masks.json"), 'w'))
        return
    
    # ── Detector stage ────────────────────────────────────────────────────────
    detector_backend = build_detector(
        args.detector, device,
        box_thr=args.box_threshold, text_thr=args.text_threshold,
        model_id=args.gdino_model,
    )

    sam2_model     = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "_ckpts/sam2.1_hiera_large.pt", device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    log.info("SAM2 image predictor initialized")

    formatted_events, all_masks = [], {}
    for i, ev in enumerate(raw_events):
        required = ("frame_index", "type", "parent_object", "child_object", "action", "parent_object_loc", "child_object_loc")
        if not all(k in ev for k in required):
            log.warning(f"Event {i} missing required fields, skipping: {ev}")
            continue
        if ev["type"] in ("assembly", "insertion", "merge"):
            ev["type"] = "composition"
        if ev["type"] not in ("decomposition", "composition"):
            log.warning(f"Event {i} unknown type '{ev['type']}', skipping")
            continue

        vlm_idx      = max(0, min(int(ev["frame_index"]), len(sample_indices) - 1))
        actual_frame = int(sample_indices[vlm_idx])
        parent_name  = ev["parent_object"]
        child_name   = ev["child_object"]
        ev_type      = ev["type"]
        vlm_parent_loc   = _process_vlm_box(ev["parent_object_loc"], W, H)
        vlm_child_loc    = _process_vlm_box(ev["child_object_loc"], W, H)
        log.info(f"Event {i}: frame={actual_frame}, type={ev_type}, "
                f"parent='{parent_name}', child='{child_name}'")

        parent_id, child_id = f"BL_{i}_parent", f"BL_{i}_child"
        all_masks[parent_id] = {}
        all_masks[child_id]  = {}
        parent_locs, child_locs = {}, {}
        
        # Context window around event frame
        ctx_start = max(0, actual_frame - args.context_frames)
        ctx_end   = min(total_frames - 1, actual_frame + args.context_frames)

        for ctx_frame, is_before in zip([ctx_start, ctx_end], [True, False]):
            if is_before:
                need_child = ev_type == "composition"
            else:
                need_child = ev_type == "decomposition"
        
            image_pil  = Image.open(frame_paths[ctx_frame]).convert("RGB")
            sam2_predictor.set_image(np.array(image_pil))

            # SAM 3 returns its own mask (4th element); GDINO only gives a box, so SAM2 segments it
            parent_det = sorted(detector_backend.detect(image_pil, parent_name), key=lambda x: x[1])[-1]
            parent_loc = parent_det[2]
            parent_locs[str(ctx_frame)] = parent_loc
            all_masks[parent_id][str(ctx_frame)] = bmask_to_rle(parent_det[3]) if len(parent_det) > 3 else _get_sam2_mask(sam2_predictor, parent_loc)

            if need_child:
                child_det = sorted(detector_backend.detect(image_pil, child_name), key=lambda x: x[1])[-1]
                child_loc = child_det[2]
                child_locs[str(ctx_frame)]  = child_loc
                all_masks[child_id][str(ctx_frame)]  = bmask_to_rle(child_det[3]) if len(child_det) > 3 else _get_sam2_mask(sam2_predictor, child_loc)

        if ev_type == "decomposition":
            start_state, end_state = [parent_id], [parent_id, child_id]
        else:
            start_state, end_state = [parent_id, child_id], [parent_id]

        formatted_events.append({
            "frame":               actual_frame,
            "type":                ev_type,
            "start_state_objects": start_state,
            "end_state_objects":   end_state,
            "vlm_parent_name":     parent_name,
            "vlm_child_name":      child_name,
            "vlm_parent_loc":      vlm_parent_loc,
            "vlm_child_loc":       vlm_child_loc,
            "det_parent_loc":      parent_locs,
            "det_child_loc":       child_locs,
            "vlm_action":          ev["action"],
            "is_candidate":        False,
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    events_path = osp.join(out_dir, f"{instance}_events.json")
    pairs_path  = osp.join(out_dir, f"{instance}_pairs.json")
    masks_path  = osp.join(out_dir, f"{instance}_final_masks.json")
    json.dump(formatted_events, open(events_path, 'w'), indent=2)
    json.dump([], open(pairs_path, 'w'))
    json.dump(all_masks, open(masks_path, 'w'))
    log.info(f"Wrote {len(formatted_events)} event(s) → {events_path}")
    log.info(f"Wrote masks for {len(all_masks)} object(s) → {masks_path}")

    if args.vis:
        vis_dir = osp.join(cfg.paths.visdir, baseline_name, instance)
        visualize_events(formatted_events, all_masks, frame_paths, vis_dir)


if __name__ == '__main__':
    main()
