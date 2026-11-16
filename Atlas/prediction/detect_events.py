import argparse
import time
import glob
import json
import os
import os.path as osp
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file, rle_to_bmask, load_tracks, save_timing


def union_pair_masks(mask_a, mask_b):
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return rle_to_bmask(mask_b)
    if mask_b is None:
        return rle_to_bmask(mask_a)

    bmask_a = rle_to_bmask(mask_a)
    bmask_b = rle_to_bmask(mask_b)
    combined = bmask_a | bmask_b
    return combined if combined.any() else None

def get_overlap_metrics(rle_emergent, rle_existing):
    if not rle_emergent or not rle_existing:
        return 0.0, 0.0

    mask_emergent = rle_to_bmask(rle_emergent)
    mask_existing = rle_to_bmask(rle_existing)
    area_emergent = mask_emergent.sum()
    area_existing = mask_existing.sum()

    if area_emergent == 0 or area_existing == 0:
        return 0.0, 0.0

    intersection = np.logical_and(mask_emergent, mask_existing).sum()
    containment = intersection / area_emergent
    iou = intersection / (area_emergent + area_existing - intersection)
    return containment, iou

def _worker_step1(fwd_id, fwd_info, rev_meta, fwd_tracks, rev_tracks, iou_thresh):
    """Processes 1-to-1 matching for a single forward track against all reverse tracks."""
    local_pairs = []
    for rev_id, rev_info in rev_meta.items():
        shared_frames = fwd_info['frames'].intersection(rev_info['frames'])
        if not shared_frames:
            continue

        total_intersection, total_union = 0, 0
        for frame_idx in shared_frames:
            mask_fwd = rle_to_bmask(fwd_tracks[frame_idx][fwd_id])
            mask_rev = rle_to_bmask(rev_tracks[frame_idx][rev_id])

            total_intersection += np.logical_and(mask_fwd, mask_rev).sum()
            total_union += np.logical_or(mask_fwd, mask_rev).sum()

        track_iou = total_intersection / total_union if total_union > 0 else 0.0
        if track_iou > iou_thresh:
            local_pairs.append((track_iou, f"F_{fwd_id}", f"R_{rev_id}"))
    return local_pairs

def _worker_step2_fwd(fwd_id, fwd_info, rev_tracks, fwd_tracks, matching_contain_thresh, matching_iou_thresh, search_window):
    """Processes forward emergence (decomposition) for a single forward track."""
    start_frame = min(fwd_info['frames'])
    if start_frame == 0:
        return [] # Track existed from the beginning

    fwd_mask = fwd_tracks[start_frame][fwd_id]
    fwd_node = f"F_{fwd_id}"
    local_events = []

    # A forward tubelet that appears after frame 0 may be a decomposition
    # product. Look for a reverse tubelet at the same frame that contains it,
    # then search backward to find the earlier forward tubelet it came from.
    for rev_id, rev_mask in rev_tracks[start_frame].items():
        containment, iou = get_overlap_metrics(fwd_mask, rev_mask)
        if containment > matching_contain_thresh and iou > matching_iou_thresh:
            for offset in range(1, search_window + 1):
                prev_frame = start_frame - offset
                if prev_frame < 0: break
                if prev_frame in rev_tracks and rev_id in rev_tracks[prev_frame]:
                    prev_rev_mask = rev_tracks[prev_frame][rev_id]
                    if prev_frame in fwd_tracks:
                        for parent_fwd_id, parent_fwd_mask in fwd_tracks[prev_frame].items():
                            parent_contain, parent_iou = get_overlap_metrics(prev_rev_mask, parent_fwd_mask)

                            if parent_contain > matching_contain_thresh and parent_iou > matching_iou_thresh:
                                rev_node = f"R_{rev_id}"
                                parent_fwd_node = f"F_{parent_fwd_id}"

                                # Keep the matching reverse node so merge_tracks can mark
                                # the forward child and reverse tubelet as duplicates, describing the
                                # same post-transformation object, then return the decomposition event.
                                local_events.append({
                                    'type': 'decomposition',
                                    'frame': start_frame,
                                    'child': fwd_node,
                                    'parent': parent_fwd_node,
                                    'rev_node': rev_node
                                })
                                return local_events
    return local_events

def _worker_step2_rev(rev_id, rev_info, rev_tracks, fwd_tracks, matching_contain_thresh, matching_iou_thresh, search_window, max_frame):
    """Processes reverse emergence (composition) for a single reverse track."""
    end_frame = max(rev_info['frames'])
    if end_frame == max_frame:
        return [] # Track existed until the very end

    rev_mask = rev_tracks[end_frame][rev_id]
    rev_node = f"R_{rev_id}"
    local_events = []

    # A reverse tubelet that appears before the final frame may indicate an
    # assembly when viewed in forward time. Match it to the forward tubelet at
    # this frame, then search forward for the later reverse tubelet it merges into.
    for fwd_id, fwd_mask in fwd_tracks[end_frame].items():
        containment, iou = get_overlap_metrics(rev_mask, fwd_mask)
        if containment > matching_contain_thresh and iou > matching_iou_thresh:
            for offset in range(1, search_window + 1):
                next_frame = end_frame + offset
                if next_frame in fwd_tracks and fwd_id in fwd_tracks[next_frame]:
                    next_fwd_mask = fwd_tracks[next_frame][fwd_id]
                    if next_frame in rev_tracks:
                        for parent_rev_id, parent_rev_mask in rev_tracks[next_frame].items():
                            parent_contain, parent_iou = get_overlap_metrics(next_fwd_mask, parent_rev_mask)

                            if parent_contain > matching_contain_thresh and parent_iou > matching_iou_thresh:
                                fwd_node = f"F_{fwd_id}"
                                parent_rev_node = f"R_{parent_rev_id}"

                                # Keep the matching forward node so merge_tracks can mark
                                # the reverse child and forward tubelet as duplicates, describing the
                                # same pre-transformation object, then return the assembly event.
                                local_events.append({
                                    'type': 'composition',
                                    'frame': end_frame,
                                    'child': rev_node,
                                    'parent': parent_rev_node,
                                    'fwd_node': fwd_node
                                })
                                return local_events
    return local_events

def merge_tracks(fwd_tracks, fwd_meta, rev_tracks, rev_meta, iou_thresh, matching_contain_thresh, matching_iou_thresh, search_window):
    events = []
    pair_records = []
    pair_lookup = {}
    paired_nodes = set()
    next_event_id = 0

    def register_pair(fwd_node, rev_node, source, event_id=None, event_type=None):
        # Store a forward/reverse duplicate pair
        key = (fwd_node, rev_node)
        if key in pair_lookup:
            return
        pair_lookup[key] = {
            'pair': [fwd_node, rev_node],
            'source': source,
            'event_id': event_id,
            'event_type': event_type,
        }
        pair_records.append(pair_lookup[key])

    print("1-to-1 Matching...")
    potential_pairs = []

    fwd_items = sorted(fwd_meta.items(), key=lambda x: int(x[0]))
    rev_items = sorted(rev_meta.items(), key=lambda x: int(x[0]))
    step1_results = [_worker_step1(fwd_id, f_info, rev_meta, fwd_tracks, rev_tracks, iou_thresh) for fwd_id, f_info in fwd_items]
    for result in step1_results:
        potential_pairs.extend(result)

    fwd_nodes = sorted({fwd_node for _, fwd_node, _ in potential_pairs})
    rev_nodes = sorted({rev_node for _, _, rev_node in potential_pairs})
    fwd_idx = {node: idx for idx, node in enumerate(fwd_nodes)}
    rev_idx = {node: idx for idx, node in enumerate(rev_nodes)}
    score_matrix = np.zeros((len(fwd_nodes), len(rev_nodes)), dtype=float)

    for track_iou, fwd_node, rev_node in potential_pairs:
        score_matrix[fwd_idx[fwd_node], rev_idx[rev_node]] = max(score_matrix[fwd_idx[fwd_node], rev_idx[rev_node]], track_iou)

    selected_pairs = []

    # Each score is the IoU between one forward and one reverse tubelet, computed
    # only over frames where both tubelets exist by accumulating IoU across those
    # shared frames, then apply Hungarian matching so each tubelet is matched at most
    # once.
    if score_matrix.size:
        row_idxs, col_idxs = linear_sum_assignment(score_matrix, maximize=True)
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            track_iou = score_matrix[row_idx, col_idx]
            if track_iou > iou_thresh:
                selected_pairs.append((track_iou, fwd_nodes[row_idx], rev_nodes[col_idx]))

    # Register 1-1 matches
    selected_pairs.sort(key=lambda x: x[0], reverse=True)
    for track_iou, fwd_node, rev_node in selected_pairs:
        if fwd_node not in paired_nodes and rev_node not in paired_nodes:
            register_pair(fwd_node, rev_node, source='iou_match')
            paired_nodes.update([fwd_node, rev_node])

    print("Detecting Transformations...")

    # Submit Forward Emergence -> Decomposition
    fwd_results = [_worker_step2_fwd(fwd_id, f_info, rev_tracks, fwd_tracks, matching_contain_thresh, matching_iou_thresh, search_window) for fwd_id, f_info in fwd_items]

    fwd_events = []
    for result in fwd_results:
        fwd_events.extend(result)

    fwd_events.sort(key=lambda ev: (ev['frame'], int(ev['child'].split('_')[1]), int(ev['rev_node'].split('_')[1]), int(ev['parent'].split('_')[1])))

    for event in fwd_events:
        event['event_id'] = next_event_id
        next_event_id += 1

        # The worker kept rev_node only to tell us which reverse tubelet is
        # the duplicate of the new forward child. Store that pair if neither
        # tubelet has already been matched elsewhere.
        fwd_node = event['child']
        rev_node = event['rev_node']

        if fwd_node not in paired_nodes and rev_node not in paired_nodes:
            register_pair(fwd_node, rev_node, source='event', event_id=event['event_id'], event_type=event['type'])
            paired_nodes.update([fwd_node, rev_node])

        # Remove the extra key before adding to final events list
        del event['rev_node']
        events.append(event)

    # Submit Reverse Emergence -> Assembly
    max_frame = max(fwd_tracks.keys()) if fwd_tracks else 0
    rev_results = [_worker_step2_rev(rev_id, r_info, rev_tracks, fwd_tracks, matching_contain_thresh, matching_iou_thresh, search_window, max_frame) for rev_id, r_info in rev_items]

    rev_events = []
    for result in rev_results:
        rev_events.extend(result)

    rev_events.sort(key=lambda ev: (ev['frame'], int(ev['child'].split('_')[1]), int(ev['fwd_node'].split('_')[1]), int(ev['parent'].split('_')[1])))

    for event in rev_events:
        event['event_id'] = next_event_id
        next_event_id += 1

        # The worker kept fwd_node only to tell us which forward tubelet is
        # the duplicate of the new reverse child. Store that pair if neither
        # tubelet has already been matched elsewhere.
        rev_node = event['child']
        fwd_node = event['fwd_node']

        if fwd_node not in paired_nodes and rev_node not in paired_nodes:
            register_pair(fwd_node, rev_node, source='event', event_id=event['event_id'], event_type=event['type'])
            paired_nodes.update([fwd_node, rev_node])

        # Remove the extra key before adding to final events list
        del event['fwd_node']
        events.append(event)

    return events, pair_records

def main():
    parser = argparse.ArgumentParser(description="Merge forward/reverse tubelets into candidate transformation events.")
    parser.add_argument('-c', "--config", default="configs/default.yaml")
    parser.add_argument('-d', '--dataset', default='vost')
    parser.add_argument('-i', '--instance', required=True)
    parser.add_argument('--track_iou_thresh', type=float, default=0.75)
    parser.add_argument('--matching_iou_thresh', type=float, default=0.05)
    parser.add_argument('--matching_contain_thresh', type=float, default=0.7)
    parser.add_argument('--skip_fp', action='store_true', help='Use the skip-fp output namespace.')
    parser.add_argument('--skip_temporal_alignment', action='store_true', help='Use the skip-temporal-alignment output namespace.')
    parser.add_argument('--search_seconds', type=float, default=1.0, help='How far to trace back/ahead for the parent tubelet.')
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, args.instance, data_cfg.image_format)))
    tracks_dir = osp.join(cfg.paths.intermdir, f'tubelets_{args.dataset}_{cfg.tubelet.entity_method}')

    fwd_tracks, fwd_meta = load_tracks(tracks_dir, args.instance, is_rev_dir=False)
    rev_tracks, rev_meta = load_tracks(tracks_dir, args.instance, is_rev_dir=True, num_frames=len(frame_paths))

    start_time = time.time()
    events, pair_records = merge_tracks(
        fwd_tracks,
        fwd_meta,
        rev_tracks,
        rev_meta,
        args.track_iou_thresh,
        args.matching_contain_thresh,
        args.matching_iou_thresh,
        int(round(args.search_seconds * data_cfg.fps)),
    )
    events.sort(key=lambda event: event['frame'])
    save_timing(cfg, f'detect_events_{args.dataset}', args.instance, time.time() - start_time, len(frame_paths))

    output_dataset = args.dataset
    if args.skip_fp: output_dataset += '_skip_fp'
    if args.skip_temporal_alignment: output_dataset += '_skip_temporal_alignment'
    out_dir = osp.join(cfg.paths.outdir, output_dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, f'{args.instance}_events.json'), 'w') as f:
        json.dump(events, f)
    with open(osp.join(out_dir, f'{args.instance}_pairs.json'), 'w') as f:
        json.dump(pair_records, f)


if __name__ == "__main__":
    main()
