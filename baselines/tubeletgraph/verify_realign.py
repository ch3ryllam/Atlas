"""
TubeletGraph-Bidir + ATLAS VLM: send every TubeletGraph-Bidir candidate event through the same VLM
verification and temporal realignment as ATLAS (Atlas/prediction/prompt_vlm.py).

Rejected events are dropped; accepted events get the VLM's frame, object names, and action.
Object masks are kept exactly as TubeletGraph-Bidir produced them.

Usage:
python3 baselines/tubeletgraph/verify_realign.py -c configs/default.yaml -d ours -s val
python3 baselines/tubeletgraph/verify_realign.py -c configs/default.yaml -d ours -i cut_orange
"""

import argparse
import glob
import json
import os
import os.path as osp
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio.v3 as imageio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, rle_to_bmask
from baselines.backends.vlm import build_vlm_client
from Atlas.prediction.html_writer import HTMLWriter
from Atlas.prediction.prompt_vlm import (get_system_prompt, get_message_prompts, get_user_content,
                                         parse_vlm_tuple, get_masked_image, fade_background, crop_candidate_image)


def split_parent_child(ev):
    """Bidir events are in forward time: decomposition [P] -> [P, C], composition [P, C] -> [P]."""
    if ev['type'] == 'decomposition':
        parent = ev['start_state_objects'][0]
        child = next(o for o in ev['end_state_objects'] if o != parent)
    else:
        parent = ev['end_state_objects'][0]
        child = next(o for o in ev['start_state_objects'] if o != parent)
    return parent, child


def process_instance(instance, cfg, data_cfg, args, in_dir, out_dir):
    model_name = cfg.vlm.model_name
    client = build_vlm_client(model_name)
    margin = int(args.context_seconds * data_cfg.fps)
    short_margin = int(1.0 * data_cfg.fps)  # 1 second look-back/look-forward
    init_color = np.array(cfg.vlm.init_color_rgb, dtype=float)
    query_color = np.array(cfg.vlm.query_color_rgb, dtype=float)
    system_prompt = get_system_prompt(skip_fp=args.skip_fp)
    prompt_decomp, prompt_comp = get_message_prompts(cfg.vlm.init_color_name, cfg.vlm.query_color_name,
                                                     skip_fp=args.skip_fp, skip_temporal_alignment=args.skip_temporal_alignment)

    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, instance, data_cfg.image_format)))
    total_frames = len(frame_paths)
    context_indices = np.linspace(0, total_frames - 1, min(args.num_context_frames, total_frames), dtype=int)

    events = json.load(open(osp.join(in_dir, f'{instance}_events.json')))
    masks = json.load(open(osp.join(in_dir, f'{instance}_final_masks.json')))

    def get_bmask(node, frame_idx):
        rle = masks.get(node, {}).get(str(frame_idx))
        return rle_to_bmask(rle) if rle else None

    html_out_path = osp.join(out_dir, f'{instance}.html')
    html_writer = HTMLWriter(title=f"Prompt Results - {instance} - {model_name}")
    html_writer.add_heading('System-level Prompt', level=3)
    html_writer.add_text(system_prompt)

    # Same per-candidate VLM query as Atlas/prediction/prompt_vlm.py; only the mask lookup differs
    formatted_events = []
    for ev in sorted(events, key=lambda x: x['frame']):
        p_node, c_node = split_parent_child(ev)
        ev_frame = ev['frame']
        is_decomp = (ev['type'] == 'decomposition')

        if is_decomp:
            # Decomposition: show frames before emergence plus a short look-ahead
            start_frame = max(0, ev_frame - margin)
            end_frame = min(total_frames - 1, ev_frame + short_margin)
        else:
            # Composition: show a short look-back plus frames after the merge
            start_frame = max(0, ev_frame - short_margin)
            end_frame = min(total_frames - 1, ev_frame + margin)

        sampled_frames = np.linspace(start_frame, end_frame, args.num_samples, dtype=int)
        prompt_msgs = prompt_decomp if is_decomp else prompt_comp
        contents = [prompt_msgs[0]]

        # First show the VLM frames from the whole video to provide context
        for i, f_idx in enumerate(context_indices):
            img = imageio.imread(frame_paths[f_idx])
            contents.append(('text', f"Context frame {i + 1} of {len(context_indices)}: original video frame {int(f_idx)}"))
            contents.append(('image', img))

        # Then show the cropped candidate window where the event should occur
        contents.append(prompt_msgs[1])
        if not args.skip_temporal_alignment:
            contents.append(('text', f'Candidate window shown for context: original video frames {int(start_frame)} through {int(end_frame)} inclusive. The returned frame_index should be the best original video frame number, and may be outside this shown window if needed.'))

        for i, f_idx in enumerate(sampled_frames):
            img = imageio.imread(frame_paths[f_idx])
            h, w = img.shape[:2]
            child_bmask = get_bmask(c_node, f_idx)
            if child_bmask is None: child_bmask = np.zeros((h, w), dtype=bool)
            parent_bmask = get_bmask(p_node, f_idx)
            if parent_bmask is None: parent_bmask = np.zeros((h, w), dtype=bool)
            parent_bmask = parent_bmask & ~child_bmask
            combined_fg_mask = parent_bmask | child_bmask
            if not combined_fg_mask.any():
                continue
            # Fade background before drawing parent/child contours
            img = fade_background(img, combined_fg_mask)
            if parent_bmask.any():
                img = get_masked_image(img, parent_bmask, init_color)
            if child_bmask.any():
                img = get_masked_image(img, child_bmask, query_color)
            img = crop_candidate_image(img, combined_fg_mask)
            contents.append(('text', f"Candidate frame {i + 1} of {len(sampled_frames)}: original video frame {int(f_idx)}"))
            contents.append(('image', img))

        candidate_image_count = sum(1 for item_type, _item in contents if item_type == 'image') - len(context_indices)
        if candidate_image_count <= 0:
            print(f"[{instance}] No visible candidate frames for {ev['type']} around {ev_frame}; skipping.")
            continue

        contents.extend(prompt_msgs[2:])
        try:
            # The VLM either rejects the candidate or returns names, action verb, and optional aligned frame
            user_content = get_user_content(contents, html_writer=html_writer)
            rsp = client.query(system_prompt, user_content, temp=args.temp)
            time.sleep(1.0)  # to avoid rate limit
            html_writer.add_heading("Response", level=3)
            html_writer.add_text(rsp)

            parsed_parent, parsed_child, parsed_action, frame_index = parse_vlm_tuple(rsp, wants_frame=not args.skip_temporal_alignment)

            if not args.skip_fp:
                # Skip this event if the VLM marks the action as a false positive
                if parsed_action.lower() == 'false positive':
                    print(f"False Positive for {ev['type']} at {ev_frame}")
                    continue

            if not args.skip_temporal_alignment:
                # Use the VLM frame if valid; otherwise keep the original frame
                if 0 <= frame_index < total_frames:
                    ev['frame'] = int(frame_index)
                else:
                    print(f"[{instance}] Invalid frame index {frame_index} for {ev['type']} at {ev_frame}.")

        except Exception as e:
            print(f"[{instance}] Exception on event around {ev_frame}: {e}")
            continue

        # Keep the Bidir objects; take frame, names, and action from the VLM
        formatted_events.append({
            **ev,
            "vlm_parent_name": parsed_parent,
            "vlm_child_name": parsed_child,
            "vlm_action": parsed_action,
        })
        html_writer.save(html_out_path)
    html_writer.save(html_out_path)

    # Masks and pairs are unchanged; only the events go through the VLM
    shutil.copy(osp.join(in_dir, f'{instance}_final_masks.json'), osp.join(out_dir, f'{instance}_final_masks.json'))
    shutil.copy(osp.join(in_dir, f'{instance}_pairs.json'), osp.join(out_dir, f'{instance}_pairs.json'))
    with open(osp.join(out_dir, f'{instance}_events.json'), 'w') as f:
        json.dump(formatted_events, f, indent=4)
    return instance, len(events), len(formatted_events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/default.yaml')
    parser.add_argument('-d', '--dataset', default='ours')
    parser.add_argument('-s', '--split')
    parser.add_argument('-i', '--instance')
    parser.add_argument('--in_dir', default='_pred_out_baselines/TubeletGraph_Baseline_B')
    parser.add_argument('--out_dir', default=None, help='default: <in_dir>_<vlm model>')
    parser.add_argument('--num_workers', type=int, default=4)
    # VLM query settings; keep in sync with Atlas/prediction/prompt_vlm.py
    parser.add_argument('--context_seconds', type=int, default=4)
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=12) # samples with contours for VLM
    parser.add_argument('--num_context_frames', type=int, default=16) # context frames provided before samples
    parser.add_argument('--skip_fp', action='store_true', help='Ablation: disable VLM to filter false positives.')
    parser.add_argument('--skip_temporal_alignment', action='store_true', help='Ablation: disable VLM for temporal alignment.')
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)

    out_dir = args.out_dir or f'{args.in_dir}_{cfg.vlm.model_name}'
    if args.skip_fp: out_dir += '_skip_fp'
    if args.skip_temporal_alignment: out_dir += '_skip_temporal_alignment'
    os.makedirs(out_dir, exist_ok=True)

    if args.instance:
        instances = [args.instance]
    else:
        if not args.split:
            parser.error('one of -i/--instance or -s/--split is required')
        with open(osp.join(data_cfg.split_dir, f'{args.split}.txt'), 'r') as f:
            instances = [line.strip() for line in f if line.strip()]
    instances = [x for x in instances if not osp.exists(osp.join(out_dir, f'{x}_events.json'))]
    print(f'{len(instances)} instances to process -> {out_dir}')

    failed = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_instance, x, cfg, data_cfg, args, args.in_dir, out_dir): x for x in instances}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                instance, n_in, n_out = future.result()
                print(f'{instance}: {n_out}/{n_in} candidate events accepted')
            except Exception as exc:
                failed.append(futures[future])
                print(f'Failed {futures[future]}: {exc}')

    if failed:
        print(f'Failed instances (rerun to retry): {failed}')
        sys.exit(1)


if __name__ == '__main__':
    main()
