"""
Semantic accuracy (S_V, S_O) and the spatiotemporal/overall numerators (st_pass, overall_pass).
For each temporally matched GT transformation (matches from temploc_pr.py, --match_in):
- S_V: GPT-4.1 judges whether the predicted action description matches the GT verb.
- Spatial match: the event's pre- and post-state objects are matched one-to-one
  (Hungarian) against the predicted participants by mask IoU >= 0.5, at the nearest
  frame to the annotated boundary where all GT objects are visible.
- S_O: for spatially matched events, GPT-4.1 judges each predicted object description.
- st_pass counts GT events that are temporally and spatially matched; overall_pass
  additionally requires the verb and all object descriptions to be correct. These are
  the numerators of H_ST,P / H_ST / H_P / H in eval_omnist.py and bootstrap_omnist.py.

Usage:
python3 eval/metrics/sem_accuracy.py -p _pred_out/ours -v <video> --pad 1 --match_in matches.json --json_out out.json
"""

import argparse
import json
import os.path as osp
import sys

import numpy as np
import openai
from pycocotools import mask as MaskUtils
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))
from utils import load_yaml_file


def judge_action_sem_acc(pred_sem, gt_sem, system_prompt, prompt_question):
    """Ask the VLM whether a predicted phrase matches the GT phrase."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": prompt_question.format(gt_sem.replace("_", " "), pred_sem)}]}],
        temperature=0.0,
    )
    rsp = response.choices[0].message.content.strip()
    return int(rsp) > 0, rsp


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("-p", "--pred", type=str, required=True)
    parser.add_argument("-a", "--anno", type=str, default=None)
    parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--json_out", type=str, default=None)
    parser.add_argument("--match_in", type=str, required=True)
    return parser


def get_obj_id(obj):
    return obj["id"] if isinstance(obj, dict) else obj


def get_obj_name(obj, obj_data=None):
    if isinstance(obj, dict) and "name" in obj:
        return obj["name"]
    if obj_data is not None and "object_name" in obj_data:
        return obj_data["object_name"]
    return str(obj)


def choose_spatial_frame(tf_index, tfms, state_type, gt_masks_data, pad_frames):
    """Pick a nearby GT frame where all state objects are visible."""
    tf = tfms[tf_index]
    target_objects = tf.get(f"{state_type}_state_objects", [])
    tf_obj_ids = {str(get_obj_id(obj)) for side in ["start", "end"] for obj in tf.get(f"{side}_state_objects", [])}

    if state_type == "start":
        original = tf["start_frame"]
        left, right = original - pad_frames, original
        for k in range(tf_index - 1, -1, -1):
            prev_ids = {str(get_obj_id(obj)) for side in ["start", "end"] for obj in tfms[k].get(f"{side}_state_objects", [])}
            if tf_obj_ids & prev_ids:
                left = max(left, tfms[k]["end_frame"] + 1)
                break
        candidates = range(left, right + 1)
    else:
        original = tf["end_frame"]
        left, right = original, original + pad_frames
        for k in range(tf_index + 1, len(tfms)):
            next_ids = {str(get_obj_id(obj)) for side in ["start", "end"] for obj in tfms[k].get(f"{side}_state_objects", [])}
            if tf_obj_ids & next_ids:
                right = min(right, tfms[k]["start_frame"] - 1)
                break
        candidates = range(right, left - 1, -1)

    for frame in candidates:
        frame_data = gt_masks_data.get(str(frame), {})
        if all(str(get_obj_id(obj)) in frame_data for obj in target_objects):
            return str(frame)
    return str(original)


def compute_action_records(instance, event_matches, pred_events, tfms):
    system_prompt = "You are a highly intelligent assistant that can analyze actions in text."
    prompt_question = (
        'Given a particular action description of "{0}", is "{1}" similar to the verbs '
        "in this action? Please rate from -1 to 1, where -1 means completely unrelated, "
        '0 means ambiguous, and 1 means "{1}" captures the meaning of "{0}" or is directly '
        "in it. Brief/general descriptions should still be considered as +1. "
        "Please answer with a single integer."
    )
    records = []

    for tf_index, tf in enumerate(tqdm(tfms, desc="Computing action desc accuracy")):
        tf_id = f"{instance}[tf={tf_index}]"
        matched_frame = event_matches.get(str(tf_index))
        pred_event = next((ev for ev in pred_events if ev.get("frame") == matched_frame and ev.get("type") == tf.get("type")), None)
        if pred_event is None:
            continue

        pred_action_desc = pred_event.get("vlm_action", pred_event.get("description"))
        score, rsp = judge_action_sem_acc(pred_action_desc, tf["description"], system_prompt, prompt_question)
        records.append({"tf_id": tf_id, "gt_action": tf["description"], "pred_action": pred_action_desc, "response": rsp, "score": score})

    return records


def matched_event_objects(matched_ev):
    if not matched_ev:
        return {}, {"start": [], "end": []}

    valid_pred_obj_ids = {"start": [str(x) for x in matched_ev.get("start_state_objects", [])], "end": [str(x) for x in matched_ev.get("end_state_objects", [])]}
    is_decomp = matched_ev.get("type") == "decomposition"
    if not matched_ev.get("start_state_objects"):
        return {}, valid_pred_obj_ids

    if is_decomp:
        parent_pred_obj_id = str(matched_ev["start_state_objects"][0])
        child_pred_obj_ids = [str(x) for x in matched_ev.get("end_state_objects", []) if str(x) != parent_pred_obj_id]
    else:
        parent_pred_obj_id = matched_ev.get("end_state_objects", [None])[0]
        parent_pred_obj_id = str(parent_pred_obj_id) if parent_pred_obj_id is not None else None
        child_pred_obj_ids = [str(x) for x in matched_ev.get("start_state_objects", []) if str(x) != parent_pred_obj_id]

    pred_obj_info = {}
    if parent_pred_obj_id is not None:
        pred_obj_info[parent_pred_obj_id] = matched_ev.get("vlm_parent_name")
    if child_pred_obj_ids:
        pred_obj_info[child_pred_obj_ids[0]] = matched_ev.get("vlm_child_name")
    return pred_obj_info, valid_pred_obj_ids


def pred_obj_id_allowed_for_state(pred_obj_id, valid_pred_obj_ids, pred_pairs):
    if str(pred_obj_id) in valid_pred_obj_ids:
        return True
    if not str(pred_obj_id).startswith("Pair_"):
        return False
    try:
        pair_idx = int(str(pred_obj_id).split("_")[1])
    except ValueError:
        return False
    return pair_idx < len(pred_pairs) and bool(set(str(x) for x in pred_pairs[pair_idx]) & valid_pred_obj_ids)


def describe_matched_pred_obj_id(matched_pred_obj_id, state_pred_obj_ids, pred_pairs, pred_obj_info):
    check_pred_obj_ids = [str(matched_pred_obj_id)]
    if str(matched_pred_obj_id).startswith("Pair_"):
        try:
            pair_idx = int(str(matched_pred_obj_id).split("_")[1])
            if pair_idx < len(pred_pairs):
                check_pred_obj_ids.extend(str(x) for x in pred_pairs[pair_idx])
        except ValueError:
            pass
    state_pred_obj_ids = set(str(x) for x in state_pred_obj_ids)
    return next((pred_obj_info[pred_obj_id] for pred_obj_id in check_pred_obj_ids if pred_obj_id in state_pred_obj_ids and pred_obj_id in pred_obj_info), None)


def compute_object_records(args, instance, event_matches, pred_events, pred_pairs, pred_obj_masks, tfms, gt_masks_data):
    system_prompt = "You are a highly intelligent assistant that can analyze actions and resulting objects in text."
    prompt_question = (
        'Given the object description "{0}", is "{1}" similar to it? '
        "Please rate from -1 to 1, where -1 means completely unrelated, "
        '0 means ambiguous, and 1 means "{1}" is similar. '
        "Over- or under-specified descriptions should still be considered as +1. "
        "Please answer with a single integer."
    )
    cfg = load_yaml_file(args.config)
    records = []
    pad_frames = args.pad * round(cfg.datasets.ours.fps * 0.5)

    for tf_index, tf in enumerate(tqdm(tfms, desc="Computing object desc accuracy")):
        tf_id = f"{instance}[tf={tf_index}]"
        matched_frame = event_matches.get(str(tf_index))
        matched_ev = next((ev for ev in pred_events if ev.get("frame") == matched_frame and ev.get("type") == tf.get("type")), None)
        pred_obj_info, valid_pred_obj_ids_per_state = matched_event_objects(matched_ev)
        gt_obj_desc_all, pred_obj_desc_all, gt_ious_all, pred_ious_all = [], [], [], []

        for state_type in ["start", "end"]:
            target_objects = tf.get(f"{state_type}_state_objects", [])
            if not target_objects:
                continue

            target_frame = choose_spatial_frame(tf_index=tf_index, tfms=tfms, state_type=state_type, gt_masks_data=gt_masks_data, pad_frames=pad_frames)
            if target_frame not in gt_masks_data:
                gt_obj_desc_all.extend(get_obj_name(obj) for obj in target_objects)
                pred_obj_desc_all.extend([None] * len(target_objects))
                gt_ious_all.extend([0.0] * len(target_objects))
                continue

            gt_obj_desc, gt_rles = [], []
            for obj in target_objects:
                obj_id = str(get_obj_id(obj))
                obj_data = gt_masks_data[target_frame].get(obj_id)
                gt_obj_desc.append(get_obj_name(obj, obj_data))
                gt_rles.append({"size": obj_data["size"], "counts": obj_data["counts"]} if obj_data is not None else None)

            valid_pred_obj_ids = set(str(x) for x in valid_pred_obj_ids_per_state[state_type])
            pred_ids, pred_rles = [], []
            for pred_obj_id, frames in pred_obj_masks.items():
                if pred_obj_id_allowed_for_state(pred_obj_id, valid_pred_obj_ids, pred_pairs) and target_frame in frames:
                    pred_ids.append(pred_obj_id)
                    pred_rles.append({"size": frames[target_frame]["size"], "counts": frames[target_frame]["counts"]})

            pred_obj_desc = [None] * len(gt_obj_desc)
            gt_ious = [0.0] * len(gt_obj_desc)
            pred_ious = [0.0] * len(pred_rles)
            valid_gt_indices = [i for i, rle in enumerate(gt_rles) if rle is not None]

            if valid_gt_indices and pred_rles:
                iou_matrix = np.zeros((len(valid_gt_indices), len(pred_rles)))
                for row_i, gt_i in enumerate(valid_gt_indices):
                    for j, pred_rle in enumerate(pred_rles):
                        iou_matrix[row_i, j] = float(MaskUtils.iou([gt_rles[gt_i]], [pred_rle], [0])[0][0])

                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                for row_i, j in zip(row_ind, col_ind):
                    gt_i = valid_gt_indices[row_i]
                    gt_ious[gt_i] = pred_ious[j] = float(iou_matrix[row_i, j])
                    if gt_ious[gt_i] >= 0.5:
                        pred_obj_desc[gt_i] = describe_matched_pred_obj_id(pred_ids[j], valid_pred_obj_ids_per_state[state_type], pred_pairs, pred_obj_info)

            gt_obj_desc_all.extend(gt_obj_desc)
            pred_obj_desc_all.extend(pred_obj_desc)
            gt_ious_all.extend(gt_ious)
            pred_ious_all.extend(pred_ious)

        scores, responses = [], []
        for gt_desc, pred_desc in zip(gt_obj_desc_all, pred_obj_desc_all):
            score, rsp = judge_action_sem_acc(pred_desc, gt_desc, system_prompt, prompt_question) if pred_desc is not None else (None, None)
            scores.append(score)
            responses.append(rsp)

        records.append({"tf_id": tf_id, "gt_ious": gt_ious_all, "pred_ious": pred_ious_all, "scores": scores, "responses": responses})

    return records


def add_stat(stats, tf_type, key, value=1):
    stats["overall"][key] += value
    if tf_type in stats:
        stats[tf_type][key] += value


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    anno_dir = args.anno or cfg.datasets.ours.anno_dir
    instance = args.video

    with open(args.match_in, "r") as f:
        event_matches = json.load(f)
    with open(osp.join(args.pred, f"{instance}_events.json"), "r") as f:
        pred_events = json.load(f)
    with open(osp.join(args.pred, f"{instance}_pairs.json"), "r") as f:
        pred_pairs = json.load(f)
    with open(osp.join(args.pred, f"{instance}_final_masks.json"), "r") as f:
        pred_obj_masks = json.load(f)
    with open(osp.join(anno_dir, f"{instance}_transformations.json"), "r") as f:
        tfms = json.load(f)
    with open(osp.join(anno_dir, f"{instance}_annotations.json"), "r") as f:
        gt_masks_data = json.load(f)

    tf_types = {f"{instance}[tf={i}]": tf.get("type", "unknown") for i, tf in enumerate(tfms)}
    action_records = compute_action_records(instance, event_matches, pred_events, tfms)
    object_records = compute_object_records(args, instance, event_matches, pred_events, pred_pairs, pred_obj_masks, tfms, gt_masks_data)

    temporal_matched = {f"{instance}[tf={k}]" for k in event_matches.keys()}
    action_sem_matched = {r["tf_id"] for r in action_records if r["score"] is True}
    spatial_matched = {r["tf_id"] for r in object_records if r["gt_ious"] and r["pred_ious"] and all(iou >= 0.5 for iou in r["gt_ious"]) and all(iou >= 0.5 for iou in r["pred_ious"])}
    object_sem_matched = {r["tf_id"] for r in object_records if all(score is True for score in r["scores"])}
    spatiotemporal_match_set = temporal_matched & spatial_matched
    overall_match_set = spatiotemporal_match_set & action_sem_matched & object_sem_matched

    stats = {cat: {"action_num": 0, "action_den": 0, "obj_num": 0, "obj_den": 0, "st_pass": 0, "overall_pass": 0, "total_gt": 0} for cat in ["overall", "decomposition", "composition"]}

    for tf_id, tf_type in tf_types.items():
        add_stat(stats, tf_type, "total_gt")
        if tf_id in spatiotemporal_match_set:
            add_stat(stats, tf_type, "st_pass")
        if tf_id in overall_match_set:
            add_stat(stats, tf_type, "overall_pass")

    for record in action_records:
        val = 1 if record["score"] is True else 0
        add_stat(stats, tf_types.get(record["tf_id"]), "action_num", val)
        add_stat(stats, tf_types.get(record["tf_id"]), "action_den")

    for record in object_records:
        spatial_ok = record["gt_ious"] and record["pred_ious"] and all(iou >= 0.5 for iou in record["gt_ious"]) and all(iou >= 0.5 for iou in record["pred_ious"])
        if not spatial_ok:
            continue
        for score in record["scores"]:
            add_stat(stats, tf_types.get(record["tf_id"]), "obj_num", 1 if score is True else 0)
            add_stat(stats, tf_types.get(record["tf_id"]), "obj_den")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(stats, f, indent=4)
