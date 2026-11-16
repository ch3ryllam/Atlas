"""
Draw the state graph for one instance: a gallery of each object's first masked frame to the
left of the divider, and a timeline of their lifetimes to the right. One lane per object; a
black arrow marks a transformation between two lanes, and an arrow in the object's own colour
marks it entering the scene or leaving for good.

Usage:
python3 Atlas/vis/render_pdf.py -c configs/default.yaml -d ours -i crackers
python3 Atlas/vis/render_pdf.py -c configs/default.yaml -d ours -i crackers --all_objects
"""
import argparse
import glob
import json
import math
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from reportlab.lib.colors import Color, black, gray
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import apply_anno, load_yaml_file, rle_to_bmask
from Atlas.vis.render_utils import build_node_to_obj_id, build_obj_colors, get_focus_obj_ids

PAGE_W, DIVIDER_X, TIMELINE_X, RIGHT_X = 2400, 800, 950, 2300
GRID_COLS, GRID_COL_W, BASE_ROW_H = 3, 250, 250
LANE_LABEL_W = TIMELINE_X - 15 - (DIVIDER_X + 10) 
GRID_FONT, GRID_LEADING = 18, 22
LANE_FONT, LANE_LEADING = 14, 17
THUMB_SIZE = 200


def lane_order(obj_id):
    prefix, _, suffix = str(obj_id).partition("_")
    return ({"Pair": 0, "F": 1, "R": 2}.get(prefix, 3),
            int(suffix) if suffix.isdigit() else 0, str(obj_id))


def wrap_lines(text, size, max_width):
    lines, current = [], ""
    for word in str(text).split():
        candidate = f"{current} {word}".strip()
        if current and stringWidth(candidate, "Helvetica-Bold", size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else lines or [""]


def prepare_data(events, node_to_obj_id, final_masks, focus_obj_ids, all_objects):
    node_map = dict(node_to_obj_id)
    names = {}
    starts, ends = defaultdict(list), defaultdict(list)

    for event in events:
        before_nodes, after_nodes = event["start_state_objects"], event["end_state_objects"]
        assert len(before_nodes) + len(after_nodes) == 3, \
            f"unexpected event shape at frame {event['frame']}"
        for node in before_nodes + after_nodes:
            node_map.setdefault(node, node)
        for role in ("parent", "child"):
            names.setdefault(node_map[event[role]], event[f"{role}_desc"])
        before = {node_map[n] for n in before_nodes}
        after = {node_map[n] for n in after_nodes}
        for obj_id in after - before:
            starts[obj_id].append(event["frame"])
        for obj_id in before - after:
            ends[obj_id].append(event["frame"])

    focus = {str(obj_id) for obj_id in (focus_obj_ids or ())}
    if not focus and not all_objects:
        print("No transformations detected: rendering every object.")

    entities = {}
    for obj_id, masks_by_frame in final_masks.items():
        if focus and not all_objects and str(obj_id) not in focus:
            continue
        visible = []
        for frame, rle in masks_by_frame.items():
            if rle:
                mask = rle_to_bmask(rle)
                if mask is not None and np.any(mask):
                    visible.append(frame)
        if not visible:
            continue
        start, end = min(visible), max(visible)
        if starts.get(obj_id):
            start = max(start, min(starts[obj_id]))
        if ends.get(obj_id):
            end = min(end, min(ends[obj_id]))
        if start > end:
            start = end = min(starts.get(obj_id, ends.get(obj_id, [min(visible)])))
        entities[obj_id] = {"start": start, "end": end, "thumbnail_frame": min(visible)}
    return node_map, names, entities, starts, ends


def make_thumbnail(frame_path, rle, color):
    mask = rle_to_bmask(rle)
    if mask is None or not np.any(mask):
        return None
    with Image.open(frame_path) as image:
        frame = np.asarray(image.convert("RGB"))
    annotated = apply_anno(frame.copy(), mask=mask, mask_color=color)
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    half = max(int(max(np.ptp(xs), np.ptp(ys)) * 1.5), 100) // 2
    h, w = annotated.shape[:2]
    crop = annotated[max(0, cy - half):min(h, cy + half),
                     max(0, cx - half):min(w, cx + half)]
    return Image.fromarray(crop) if crop.size else None


def draw_gallery(pdf, obj_ids, entities, grid_labels, row_h, final_masks,
                 frame_paths, colors, page_h):
    for idx, obj_id in enumerate(tqdm(obj_ids, desc="Drawing thumbnails")):
        row, col = divmod(idx, GRID_COLS)
        x, y = 50 + col * GRID_COL_W, page_h - 100 - (row + 1) * row_h
        lines = grid_labels[obj_id]
        caption_top = y + row_h - 20
        pdf.setFont("Helvetica-Bold", GRID_FONT)
        pdf.setFillColor(black)
        for i, line in enumerate(lines):
            pdf.drawString(x, caption_top - i * GRID_LEADING, line)
        frame = entities[obj_id]["thumbnail_frame"]
        if not 0 <= frame < len(frame_paths):
            continue
        thumbnail = make_thumbnail(frame_paths[frame], final_masks[obj_id][frame], colors[obj_id])
        if thumbnail is None:
            continue
        scale = min(THUMB_SIZE / thumbnail.width, THUMB_SIZE / thumbnail.height)
        w, h = thumbnail.width * scale, thumbnail.height * scale
        caption_bottom = caption_top - (len(lines) - 1) * GRID_LEADING
        pdf.drawImage(ImageReader(thumbnail), x, caption_bottom - 10 - h, width=w, height=h)


def draw_arrow(pdf, x, y_from, y_to):
    pdf.line(x, y_from, x, y_to)
    direction = 1 if y_to > y_from else -1
    pdf.line(x, y_to, x - 5, y_to - direction * 8)
    pdf.line(x, y_to, x + 5, y_to - direction * 8)


def draw_timeline(pdf, obj_ids, entities, events, node_map, starts, ends,
                  lane_labels, colors, frame_paths, fps, page_h):
    duration = max(1, math.ceil((len(frame_paths) - 1) / fps))
    def frame_x(frame):
        return TIMELINE_X + (frame / fps / duration) * (RIGHT_X - TIMELINE_X)
    lane_y = {
        obj_id: page_h - 100 - idx * (page_h - 200) / max(1, len(obj_ids) - 1)
        for idx, obj_id in enumerate(obj_ids)
    }
    pdf.setStrokeColor(black)
    pdf.setLineWidth(1)
    pdf.line(TIMELINE_X, 100, RIGHT_X, 100)
    for second in range(duration + 1):
        x = frame_x(second * fps)
        pdf.setStrokeColor(gray)
        pdf.setDash(2, 4)
        pdf.line(x, 100, x, page_h - 100)
        pdf.setDash()
        pdf.setStrokeColor(black)
        pdf.line(x, 100, x, 95)
        pdf.setFont("Helvetica", 14)
        pdf.drawCentredString(x, 80, f"{second}s")
        pdf.setFont("Helvetica-Oblique", 11)
        pdf.drawCentredString(x, 65, f"({int(round(second * fps))}f)")

    pdf.setLineWidth(2)
    for obj_id in obj_ids:
        y = lane_y[obj_id]
        start, end = entities[obj_id]["start"], entities[obj_id]["end"]
        pdf.setStrokeColor(Color(*(channel / 255 for channel in colors[obj_id])))
        if start > 0 and obj_id not in starts:
            x = frame_x(start)
            pdf.line(x, y + 20, x, y)
            pdf.line(x, y, x - 5, y + 8)
            pdf.line(x, y, x + 5, y + 8)
        if end < len(frame_paths) - 1 and obj_id not in ends:
            x = frame_x(end)
            pdf.line(x, y, x, y - 20)
            pdf.line(x, y - 20, x - 5, y - 12)
            pdf.line(x, y - 20, x + 5, y - 12)

    pdf.setStrokeColor(black)
    pdf.setLineWidth(2)
    for event in events:
        parent = node_map[event["parent"]]
        child = node_map[event["child"]]
        if parent not in lane_y or child not in lane_y:
            continue
        source, target = ((parent, child) if event["type"] == "decomposition"
                          else (child, parent))
        draw_arrow(pdf, frame_x(event["frame"]), lane_y[source], lane_y[target])

    for obj_id in obj_ids:
        y = lane_y[obj_id]
        lines = lane_labels[obj_id]
        pdf.setFont("Helvetica-Bold", LANE_FONT)
        pdf.setFillColor(black)
        first_y = y - 5 + (len(lines) - 1) * (LANE_LEADING // 2)
        for i, line in enumerate(lines):
            pdf.drawRightString(TIMELINE_X - 15, first_y - i * LANE_LEADING, line)
        color = Color(*(channel / 255 for channel in colors[obj_id]))
        pdf.setStrokeColor(color)
        pdf.setFillColor(color)
        pdf.setLineWidth(5)
        x1, x2 = frame_x(entities[obj_id]["start"]), frame_x(entities[obj_id]["end"])
        pdf.line(x1, y, x2, y)
        pdf.circle(x1, y, 5, fill=1, stroke=0)
        pdf.circle(x2, y, 5, fill=1, stroke=0)


def render_pdf(node_to_obj_id, events, frame_paths, fps, out_path,
                          final_masks, focus_obj_ids=None, all_objects=False):
    if not frame_paths or fps <= 0:
        raise ValueError("At least one frame and a positive fps are required.")
    node_map, names, entities, starts, ends = prepare_data(
        events, node_to_obj_id, final_masks, focus_obj_ids, all_objects
    )
    if not entities:
        print("No entities found to plot.")
        return
    obj_ids = sorted(entities, key=lane_order)
    colors = build_obj_colors(final_masks, focus_obj_ids or ())
    grid_labels = {o: wrap_lines(names.get(o, o), GRID_FONT, GRID_COL_W - 20) for o in obj_ids}
    lane_labels = {o: wrap_lines(names.get(o, o), LANE_FONT, LANE_LABEL_W) for o in obj_ids}
    row_h = BASE_ROW_H + (max(map(len, grid_labels.values())) - 1) * GRID_LEADING
    lane_h = max(60, max(map(len, lane_labels.values())) * LANE_LEADING + 15)
    page_h = max(math.ceil(len(obj_ids) / GRID_COLS) * row_h + 150,
                 len(obj_ids) * lane_h + 200, 600)
    pdf = canvas.Canvas(out_path, pagesize=(PAGE_W, page_h))
    pdf.setStrokeColor(black)
    pdf.setLineWidth(4)
    pdf.line(DIVIDER_X, 0, DIVIDER_X, page_h)
    draw_gallery(pdf, obj_ids, entities, grid_labels, row_h, final_masks,
                 frame_paths, colors, page_h)
    draw_timeline(pdf, obj_ids, entities, events, node_map, starts, ends,
                  lane_labels, colors, frame_paths, fps, page_h)
    pdf.save()
    print(f"State Graph saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-d", "--dataset", default="vost")
    parser.add_argument("-i", "--instance", required=True)
    parser.add_argument("--skip_fp", action="store_true")
    parser.add_argument("--skip_temporal_alignment", action="store_true")
    parser.add_argument("--all_objects", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    suffix = args.dataset
    if args.skip_fp:
        suffix += "_skip_fp"
    if args.skip_temporal_alignment:
        suffix += "_skip_temporal_alignment"
    out_dir = osp.join(cfg.paths.outdir, suffix)
    vis_dir = osp.join(cfg.paths.visdir, suffix)
    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, args.instance, data_cfg.image_format)))
    prefix = osp.join(out_dir, args.instance)
    with open(f"{prefix}_events.json") as f:
        formatted_events = json.load(f)
    with open(f"{prefix}_pairs.json") as f:
        object_groups = [tuple(group) for group in json.load(f)]
    with open(f"{prefix}_final_masks.json") as f:
        raw_masks = json.load(f)
    final_masks = {
        obj_id: {int(frame): mask for frame, mask in frames.items()}
        for obj_id, frames in raw_masks.items()
    }
    events = []
    for event in formatted_events:
        decomposition = event["type"] == "decomposition"
        before, after = event["start_state_objects"], event["end_state_objects"]
        parent = (before if decomposition else after)[0]
        child = next(n for n in (after if decomposition else before) if n != parent)
        events.append({
            "parent": parent, "child": child, "frame": event["frame"],
            "type": event["type"], "parent_desc": event["vlm_parent_name"],
            "child_desc": event["vlm_child_name"],
            "start_state_objects": before, "end_state_objects": after,
        })
    node_map = build_node_to_obj_id(object_groups)
    focus_ids = get_focus_obj_ids(events, node_map)
    os.makedirs(vis_dir, exist_ok=True)
    render_pdf(
        node_map, events, frame_paths, data_cfg.fps,
        osp.join(vis_dir, f"{args.instance}_state_graph.pdf"),
        final_masks, focus_obj_ids=focus_ids, all_objects=args.all_objects,
    )


if __name__ == "__main__":
    main()
