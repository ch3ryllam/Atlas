"""
Verify candidate transformations from detect_events.py with a VLM, and name and temporally realign the accepted ones.
Overwrites {instance}_events.json / {instance}_pairs.json in place and writes {instance}.html with every prompt and response.

Usage:
python3 Atlas/prediction/prompt_vlm.py -c configs/default.yaml -d ours -i crackers
python3 Atlas/prediction/prompt_vlm.py -c configs/default.yaml -d ours -i crackers --skip_fp --skip_temporal_alignment
"""

import ast, base64, io, os, json, glob, argparse, re, sys, time
import os.path as osp

import imageio.v3 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import load_yaml_file, rle_to_bmask, apply_anno, load_tracks
import openai
from Atlas.prediction.html_writer import HTMLWriter
from Atlas.prediction.detect_events import union_pair_masks


def encode_image_from_np(image_np, max_side=768, jpeg_quality=82):
    """Convert a numpy array (HxWxC format) to a base64 encoded JPEG, downscaled to max_side."""
    assert image_np.ndim == 3 and image_np.shape[2] in [3, 4]
    if image_np.max() <= 1.0:
        image_np = image_np * 255

    image = Image.fromarray(image_np.astype(np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Downscale so the longer side is at most max_side
    width, height = image.size
    if max_side and max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
        image = image.resize((width, height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/jpeg"


def get_image_payload(image, detail="low"):
    base64_image, mime_type = encode_image_from_np(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": detail},
    }


def get_text_payload(text):
    return {"type": "text", "text": text}


def get_user_content(data, html_writer=None, image_detail="low"):
    if html_writer is not None:
        html_writer.add_heading("Prompt", level=3)
    content = []
    for data_type, x in data:
        if data_type == "text":
            content.append(get_text_payload(x))
            if html_writer is not None:
                html_writer.add_text(x)
        elif data_type == "image":
            content.append(get_image_payload(x, image_detail))
            if html_writer is not None:
                html_writer.add_image(x)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    return content


def get_model_response(response, html_writer=None, sleep_time=0):
    time.sleep(sleep_time)  # to avoid rate limit
    rsp = response.choices[0].message.content
    if html_writer is not None:
        html_writer.add_heading("Response", level=3)
        html_writer.add_text(rsp)
    return rsp


def get_system_prompt(skip_fp=False):
    prompt = (
        "You are analyzing videos to verify physical transformation events.\n\n"
        "A transformation event is one of:\n"
        "- decomposition: one object splits, opens, separates, or reveals another distinct object\n"
        "- composition: two distinct objects combine, merge, are inserted together, or one object enters/joins another\n\n"
        "Use the whole-video context frames to understand the scene, object identities, and action. "
        "Then answer only about the candidate transformation clip with colored contours.\n\n"
        "Do not confuse camera motion, occlusion, pose change, lighting change with a transformation. "
        "A valid transformation must involve two spatially distinct objects or object parts whose relationship physically changes.\n\n"
    )
    if not skip_fp:
        prompt += (
            "For every candidate, decide whether it is a real transformation or a false positive. "
            "Use false positive when the contours are unrelated, mostly background, or do not show a real "
            "decomposition/composition event.\n\n"
        )

    prompt += "Follow the requested output format exactly. Do not include explanations."
    return prompt


def get_message_prompts(init_c_name, query_c_name, skip_fp=False, skip_temporal_alignment=False):
    wants_fp = not skip_fp
    wants_frame = not skip_temporal_alignment

    context_intro = (
        "We will first show whole-video context frames sampled uniformly across the video. "
        "These frames are unannotated and are only for understanding the scene and object identities."
    )
    candidate_intro = (
        "Now we will show a candidate transformation clip. These frames are local to one possible event.\n\n"
        "Candidate frames are zoomed crops from the original video frame. Use the full-video context frames "
        "to understand scene location, and use these crops to inspect the candidate contours.\n\n"
        "In the candidate clip:\n"
        f"- the {init_c_name} contour marks the proposed parent / continuing object\n"
        f"- the {query_c_name} contour marks the proposed child / second object\n"
        "- faded background is not important; focus on the contoured objects\n\n"
        "Each candidate image is labeled with its original video frame number. If asked for a frame index, "
        "return the original video frame number, not the displayed candidate image number."
    )

    if wants_frame:
        output_instruction = (
            'Output exactly one Python tuple:\n'
            '("object_1", "object_2", "action", frame_index)\n\n'
            'The first three entries must be strings wrapped in double quotes. '
            'For a real transformation, action must be a one-word verb. '
        )
        if wants_fp:
            output_instruction += (
                'For a false positive, action must be exactly "false positive". '
                'frame_index must be an integer original video frame number. Use frame_index 0 for false positive. '
            )
        else:
            output_instruction += 'frame_index must be an integer original video frame number. '
    else:
        output_instruction = (
            'Output exactly one Python tuple:\n'
            '("object_1", "object_2", "action")\n\n'
            'All entries must be strings wrapped in double quotes. '
            'For a real transformation, action must be a one-word verb. '
        )
        if wants_fp:
            output_instruction += 'For a false positive, action must be exactly "false positive". '
    output_instruction += 'Do not output JSON, markdown, explanation, or any extra text.'

    false_positive_instruction = ''
    if wants_fp:
        false_positive_instruction = 'If NO: provide the object descriptions as normal and set the action to exactly "false positive".'
        if wants_frame:
            false_positive_instruction += ' Output 0 for the frame index.'

    if wants_fp:
        decomp_instruction = (
            f"Question: In the candidate clip, does the {query_c_name} object physically separate from, emerge from, "
            f"open out of, or become a distinct visible part of the {init_c_name} object?\n\n"
            "Use the whole-video context only to understand what the objects are and what action is happening. "
            "Make the yes/no decision from the candidate clip.\n\n"
        )
        comp_instruction = (
            f"Question: In the candidate clip, does the {query_c_name} object physically combine with, enter, attach to, "
            f"merge into, or get inserted into the {init_c_name} object?\n\n"
            "Use the whole-video context only to understand what the objects are and what action is happening. "
            "Make the yes/no decision from the candidate clip.\n\n"
        )
    else:
        decomp_instruction = (
            f"Identify the decomposition shown in the candidate clip: the {query_c_name} object physically separates from, "
            f"emerges from, opens out of, or becomes a distinct visible part of the {init_c_name} object.\n\n"
            "Use the whole-video context only to understand what the objects are and what action is happening. "
            "Use the candidate clip to describe the contoured objects and transformation.\n\n"
        )
        comp_instruction = (
            f"Identify the composition shown in the candidate clip: the {query_c_name} object physically combines with, enters, "
            f"attaches to, merges into, or gets inserted into the {init_c_name} object.\n\n"
            "Use the whole-video context only to understand what the objects are and what action is happening. "
            "Use the candidate clip to describe the contoured objects and transformation.\n\n"
        )

    if wants_frame:
        decomp_instruction += (
            f"Describe the {init_c_name} object and the {query_c_name} object each in three words or less "
            "while including the object name. Then describe the action with one verb only. "
            "Finally, provide the integer original video frame number where the separation occurs. "
            "The displayed candidate frames are context; you may choose a frame between them or slightly outside "
            "the shown candidate window if that is more accurate.\n\n"
        )
        comp_instruction += (
            f"Describe the {init_c_name} object and the {query_c_name} object each in three words or less "
            "while including the object name. Then describe the action with one verb only. "
            "Finally, provide the integer original video frame number where the combination occurs. "
            "The displayed candidate frames are context; you may choose a frame between them or slightly outside "
            "the shown candidate window if that is more accurate.\n\n"
        )
    else:
        object_action_instruction = (
            f"Describe the {init_c_name} object and the {query_c_name} object each in three words or less "
            "while including the object name. Then describe the action with one verb only.\n\n"
        )
        decomp_instruction += object_action_instruction
        comp_instruction += object_action_instruction

    decomp_instruction += false_positive_instruction
    comp_instruction += false_positive_instruction

    prompt_decomp = [
        ('text', context_intro),
        ('text', candidate_intro),
        ('text', decomp_instruction),
        ('text', output_instruction),
    ]
    prompt_comp = [
        ('text', context_intro),
        ('text', candidate_intro),
        ('text', comp_instruction),
        ('text', output_instruction),
    ]

    return prompt_decomp, prompt_comp

def get_rle(node, frame_idx, fwd_tracks, rev_tracks):
    """Mask (RLE) of a tubelet node at a frame, or None if the node is empty or not visible there."""
    # Track nodes are stored as F_<id> or R_<id>
    if not node: return None
    prefix, nid = node.split('_')
    tracks = fwd_tracks if prefix == 'F' else rev_tracks
    return tracks.get(frame_idx, {}).get(str(nid))

def get_masked_image(image, mask, color_rgb, mask_alpha=0.0, contour_thickness=3):
    if isinstance(mask, dict): mask = rle_to_bmask(mask)
    image = apply_anno(image.copy(), mask=mask, mask_color=color_rgb, mask_alpha=mask_alpha, contour_thickness=contour_thickness)
    return image

def fade_background(image, fg_mask, alpha=0.5):
    """Fade background toward white"""
    bg_mask = ~fg_mask
    image_bg = image[bg_mask].astype(np.float32)
    image[bg_mask] = (image_bg * (1 - alpha) + 255 * alpha).astype(np.uint8)
    return image

def crop_candidate_image(image, crop_mask, pad_frac=0.5, min_crop_frac=0.45):
    img_h, img_w = image.shape[:2]
    y_idx, x_idx = np.where(crop_mask)
    x1, x2 = int(x_idx.min()), int(x_idx.max())
    y1, y2 = int(y_idx.min()), int(y_idx.max())

    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)
    pad = int(max(bbox_w, bbox_h) * pad_frac)
    crop_w = max(bbox_w + 2 * pad, int(img_w * min_crop_frac))
    crop_h = max(bbox_h + 2 * pad, int(img_h * min_crop_frac))
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    new_x1 = max(0, min(img_w - crop_w, cx - crop_w // 2))
    new_y1 = max(0, min(img_h - crop_h, cy - crop_h // 2))
    new_x2 = new_x1 + crop_w
    new_y2 = new_y1 + crop_h

    return image[new_y1:new_y2, new_x1:new_x2]


def parse_vlm_tuple(raw_rsp, wants_frame):
    """Parse the VLM's (parent, child, action[, frame_index]) tuple response."""
    match = re.search(r"\((.*?)\)", raw_rsp.replace("\n", ""))
    try:
        parsed = ast.literal_eval(f"({match.group(1)})")
    except Exception as e:
        raise ValueError(f"Could not parse tuple from response: {raw_rsp}") from e

    if not isinstance(parsed, tuple):
        raise ValueError(f"Response was not a tuple: {raw_rsp}")

    expected_len = 4 if wants_frame else 3
    if len(parsed) != expected_len:
        raise ValueError(f"Expected tuple of length {expected_len}, got {len(parsed)}: {parsed}")

    parsed_parent = str(parsed[0]).strip()
    parsed_child = str(parsed[1]).strip()
    parsed_action = str(parsed[2]).strip()
    frame_index = int(parsed[3]) if wants_frame else None
    return parsed_parent, parsed_child, parsed_action, frame_index


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", default="configs/default.yaml")
    parser.add_argument('-d', '--dataset', default='vost')
    parser.add_argument('-i', '--instance', required=True)
    parser.add_argument('--context_seconds', type=float, default=4) # omega_long
    parser.add_argument('--short_seconds', type=float, default=1.0) # omega_short
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=12) # samples with contours for VLM
    parser.add_argument('--num_context_frames', type=int, default=16) # context frames provided before samples
    parser.add_argument('--skip_fp', action='store_true', help='Ablation: disable VLM to filter false positives.')
    parser.add_argument('--skip_temporal_alignment', action='store_true', help='Ablation: disable VLM for temporal alignment.')
    return parser


def main():
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    margin = int(args.context_seconds * data_cfg.fps)
    short_margin = int(args.short_seconds * data_cfg.fps)
    instance = args.instance
    model_name = cfg.vlm.model_name
    client = openai.OpenAI()
    init_color = np.array(cfg.vlm.init_color_rgb, dtype=float)
    query_color = np.array(cfg.vlm.query_color_rgb, dtype=float)
    init_c_name = cfg.vlm.init_color_name
    query_c_name = cfg.vlm.query_color_name
    output_dataset = args.dataset
    if args.skip_fp: output_dataset += '_skip_fp'
    if args.skip_temporal_alignment: output_dataset += '_skip_temporal_alignment'

    system_prompt = get_system_prompt(skip_fp=args.skip_fp)
    pred_out_dir = osp.join(cfg.paths.outdir, output_dataset)
    os.makedirs(pred_out_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, instance, data_cfg.image_format)))
    total_frames = len(frame_paths)
    context_indices = np.linspace(0, total_frames - 1, min(args.num_context_frames, total_frames), dtype=int)

    tracks_dir = osp.join(cfg.paths.intermdir, f'tubelets_{args.dataset}_{cfg.tubelet.entity_method}')
    fwd, _ = load_tracks(tracks_dir, instance, is_rev_dir=False)
    rev, _ = load_tracks(tracks_dir, instance, is_rev_dir=True, num_frames=total_frames)

    events_path = osp.join(pred_out_dir, f'{instance}_events.json')
    pairs_path = osp.join(pred_out_dir, f'{instance}_pairs.json')
    if not osp.exists(events_path) or not osp.exists(pairs_path):
        raise FileNotFoundError('Run Atlas/prediction/detect_events.py first.')
    
    with open(events_path, 'r') as f:
        events = json.load(f)
    with open(pairs_path, 'r') as f:
        pair_records = json.load(f)

    merged_pairs = [tuple(rec['pair']) for rec in pair_records]
    pair_map = {node: other for f, r in merged_pairs for node, other in ((f, r), (r, f))}

    events.sort(key=lambda x: x['frame'])

    # Query the VLM to verify events, name objects/actions, and realign prediction
    formatted_events = []
    html_out_path = osp.join(pred_out_dir, f'{instance}.html')
    html_writer = HTMLWriter(title=f"Prompt Results - {instance} - {model_name}")
    html_writer.add_heading('System-level Prompt', level=3)
    html_writer.add_text(system_prompt)

    prompt_decomp, prompt_comp = get_message_prompts(init_c_name, query_c_name, skip_fp=args.skip_fp, skip_temporal_alignment=args.skip_temporal_alignment)

    for ev in tqdm(events, desc=f"Processing {instance}"):
        p_node, c_node, ev_frame = ev['parent'], ev['child'], ev['frame']
        is_decomp = (ev['type'] == 'decomposition')

        if is_decomp:
            # Decomposition: show frames before emergence plus a short look-ahead
            start_frame = max(0, ev_frame - margin)
            end_frame = min(total_frames - 1, ev_frame + short_margin)
        else:
            # Assembly: show a short look-back plus frames after the merge
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
            contents.append(('text', f'Candidate window shown for context: original video frames {int(start_frame)} through {int(end_frame)} inclusive. The returned frame_index should be the best original video frame number.'))

        for i, f_idx in enumerate(sampled_frames):
            img = imageio.imread(frame_paths[f_idx])
            h, w = img.shape[:2]
            child_bmask = union_pair_masks(get_rle(c_node, f_idx, fwd, rev), get_rle(pair_map.get(c_node), f_idx, fwd, rev))
            if child_bmask is None: child_bmask = np.zeros((h, w), dtype=bool)
            parent_bmask = union_pair_masks(get_rle(p_node, f_idx, fwd, rev), get_rle(pair_map.get(p_node), f_idx, fwd, rev))
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

        contents.extend(prompt_msgs[2:])
        try:
            # The VLM either rejects the candidate or returns names, action verb, and optional aligned frame
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": get_user_content(contents, html_writer=html_writer)},
                ],
                temperature=args.temp,
            )
            rsp = get_model_response(response, html_writer=html_writer, sleep_time=1.0)

            parsed_parent, parsed_child, parsed_action, frame_index = parse_vlm_tuple(rsp, wants_frame=not args.skip_temporal_alignment)

            if not args.skip_fp:
                # Skip this event if the VLM marks the action as a false positive
                if parsed_action.lower() == 'false positive':
                    print(f"False Positive for {ev['type']} at {ev_frame}")
                    continue

            if not args.skip_temporal_alignment:
                if 0 <= frame_index < total_frames:
                    ev['frame'] = int(frame_index)

        except Exception as e:
            print(f"[{instance}] Exception on event around {ev_frame}: {e}")
            continue

        start_objs, end_objs = ([p_node], [p_node, c_node]) if is_decomp else ([p_node, c_node], [p_node])

        formatted_events.append({"frame": ev['frame'], "type": ev['type'], "start_state_objects": start_objs, "end_state_objects": end_objs, "vlm_parent_name": parsed_parent, "vlm_child_name": parsed_child, "vlm_action": parsed_action})
        html_writer.save(html_out_path)

    if not events:
        html_writer.add_heading('No Candidate Events', level=3)
        html_writer.add_text('No candidate events were found for this instance.')
    elif not formatted_events:
        html_writer.add_heading('No Accepted Events', level=3)
        html_writer.add_text('Candidate events were processed, but none survived VLM filtering/parsing.')
    html_writer.save(html_out_path)

    final_pairs = [list(rec['pair']) for rec in pair_records]

    with open(events_path, 'w') as f:
        json.dump(formatted_events, f)
    with open(pairs_path, 'w') as f:
        json.dump(final_pairs, f)

if __name__ == "__main__":
    main()

