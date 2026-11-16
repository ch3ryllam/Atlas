"""
Render the side-by-side video for one instance: the original frames on the left and the same
frames with each object's mask overlaid on the right. Only the objects taking part in a
transformation are overlaid unless --all_objects is given.

Usage:
python3 Atlas/vis/render_video.py -c configs/default.yaml -d ours -i crackers
python3 Atlas/vis/render_video.py -c configs/default.yaml -d ours -i crackers --all_objects
"""
import os, json, glob, argparse, sys, concurrent.futures
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, write_video
from Atlas.vis.render_utils import overlay, build_node_to_obj_id, build_obj_colors, get_focus_obj_ids


def render_single_frame(task):
    f_path, final_masks, obj_colors = task
    base = np.array(Image.open(f_path), dtype=np.uint8)
    ol = overlay(base, final_masks, obj_colors) if final_masks else base
    return np.hstack([base, ol])


def render_video(out_path, frame_paths, fps, all_final_masks, obj_colors, transform_only=None):
    if transform_only:
        all_final_masks = {obj_id: mask for obj_id, mask in all_final_masks.items() if obj_id in transform_only}

    render_tasks = []
    for idx, f_path in enumerate(frame_paths):
        final_display_masks = {
            obj_id: mask[str(idx)]
            for obj_id, mask in all_final_masks.items()
            if str(idx) in mask
        }
        render_tasks.append((f_path, final_display_masks, obj_colors))

    vis_out = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(render_single_frame, render_tasks)
        for result in tqdm(results, total=len(render_tasks), desc="Rendering Frames"):
            vis_out.append(result)

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    write_video(out_path, vis_out, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', "--config", default="configs/default.yaml")
    p.add_argument('-d', '--dataset', default='vost')
    p.add_argument('-i', '--instance', required=True)
    p.add_argument('--skip_fp', action='store_true')
    p.add_argument('--skip_temporal_alignment', action='store_true')
    p.add_argument('--all_objects', action='store_true', help="overlay every object, not just the ones in focus")
    args = p.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)

    instance = args.instance
    output_dataset = args.dataset
    if args.skip_fp: output_dataset += '_skip_fp'
    if args.skip_temporal_alignment: output_dataset += '_skip_temporal_alignment'

    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, instance, data_cfg.image_format)))

    with open(osp.join(cfg.paths.outdir, output_dataset, f'{instance}_events.json'), 'r') as f:
        formatted_events = json.load(f)

    events = []
    for event in formatted_events:
        decomposition = event['type'] == 'decomposition'
        before, after = event['start_state_objects'], event['end_state_objects']
        parent = (before if decomposition else after)[0]
        # The child first appears at the event frame and the parent is found earlier
        # (Atlas/prediction/detect_events.py), so the two are never the same node.
        child = next(n for n in (after if decomposition else before) if n != parent)
        events.append({
            'parent': parent,
            'child': child,
            'frame': event['frame'],
            'type': event['type'],
            'start_state_objects': before,
            'end_state_objects': after,
        })

    with open(osp.join(cfg.paths.outdir, output_dataset, f'{instance}_pairs.json'), 'r') as f:
        pairs = json.load(f)

    with open(osp.join(cfg.paths.outdir, output_dataset, f'{instance}_final_masks.json'), 'r') as f:
        all_final_masks = json.load(f)

    focus_obj_ids = get_focus_obj_ids(events, build_node_to_obj_id(pairs))
    obj_colors = build_obj_colors(all_final_masks, focus_obj_ids)
    render_video(
        osp.join(cfg.paths.visdir, output_dataset, f'{instance}_final.mp4'),
        frame_paths,
        data_cfg.fps,
        all_final_masks,
        obj_colors,
        transform_only=None if args.all_objects else focus_obj_ids,
    )

if __name__ == "__main__":
    main()
