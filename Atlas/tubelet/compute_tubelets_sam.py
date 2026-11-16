import time
import json, os, random, sys, glob
import os.path as osp
import argparse
import numpy as np
import torch
from types import SimpleNamespace
from pycocotools import mask as MaskUtils

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from Atlas.tracker import get_tracker
from utils import rle_to_bmask, bmask_to_rle, coverage, load_yaml_file, save_all_to_json, load_anno, save_timing


def add_mask(sam2, obj_id, mask, all_tracks, tracked_objs, global_confs, frame_idx):
    sam2.predictor.add_new_mask(
        inference_state=sam2.inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask,
    )
    all_tracks[frame_idx][obj_id] = bmask_to_rle(mask)
    global_confs[frame_idx][obj_id] = float('inf')
    tracked_objs[obj_id] = {'mask': bmask_to_rle(mask), 'init_frame_idx': frame_idx}


def check_duplicate(out_frame_idx, obj_idx, rle, conf, all_tracks, global_confs, removed_objects, tracked_objs, duplicate_iou_thresh=0.9, min_track_age=5):
    if obj_idx in removed_objects:
        return True
    for existing_obj, existing_rle in list(all_tracks[out_frame_idx].items()):
        if existing_obj == obj_idx or existing_obj in removed_objects:
            continue
        iou = MaskUtils.iou([rle], [existing_rle], [False])[0][0]
        if iou <= duplicate_iou_thresh:
            continue
        age1 = out_frame_idx - tracked_objs[obj_idx]['init_frame_idx']
        age2 = out_frame_idx - tracked_objs[existing_obj]['init_frame_idx']
        if age1 <= min_track_age or age2 <= min_track_age:
            continue
        existing_conf = global_confs.get(out_frame_idx, {}).get(existing_obj, -float('inf'))
        if conf < existing_conf:
            removed_objects.add(obj_idx)
            print(f'[Frame {out_frame_idx}] Removed duplicate Obj {obj_idx}; IoU={iou:.2f}, conf={conf:.2f} < Obj {existing_obj} conf={existing_conf:.2f}')
            return True
        removed_objects.add(existing_obj)
        all_tracks[out_frame_idx].pop(existing_obj, None)
        global_confs[out_frame_idx].pop(existing_obj, None)
        print(f'[Frame {out_frame_idx}] Removed duplicate Obj {existing_obj}; IoU={iou:.2f}, conf={existing_conf:.2f} < Obj {obj_idx} conf={conf:.2f}')
    return False


def propagate(sam2, all_tracks, start_frame, removed_objects, global_confs, tracked_objs, stop_frame=None, clean_prev_memory=20):
    obj_ids = [x for x in sam2.inference_state['obj_idx_to_id'].values()]
    if len(obj_ids) == 0:
        print('No objects to track')
        return
    end_msg = 'end' if stop_frame is None else f'frame={stop_frame}'
    print('Tracking {} objects from id={} to id={} [frame={} -> {}]...'.format(len(obj_ids), min(obj_ids), max(obj_ids), start_frame, end_msg))

    for out_frame_idx, out_obj_ids, out_mask_logits in sam2.predictor.propagate_in_video(sam2.inference_state):
        if out_frame_idx <= start_frame:
            continue

        if out_frame_idx not in all_tracks:
            all_tracks[out_frame_idx] = dict()
        if out_frame_idx not in global_confs:
            global_confs[out_frame_idx] = dict()

        for i, obj_idx in enumerate(out_obj_ids):
            if obj_idx in removed_objects:
                continue

            mask = (out_mask_logits[i, 0] > 0.0).cpu().numpy()
            rle = bmask_to_rle(mask)
            conf = torch.sigmoid(out_mask_logits[i, 0]).mean().item()

            if check_duplicate(out_frame_idx, obj_idx, rle, conf, all_tracks, global_confs, removed_objects, tracked_objs):
                continue

            all_tracks[out_frame_idx][obj_idx] = rle
            global_confs[out_frame_idx][obj_idx] = conf

        delete_f_idx = out_frame_idx - clean_prev_memory
        for obj_idx, obj_output in sam2.inference_state['output_dict_per_obj'].items():
            if obj_idx in removed_objects:
                obj_output['non_cond_frame_outputs'].clear()
            elif delete_f_idx in obj_output['non_cond_frame_outputs']:
                obj_output['non_cond_frame_outputs'][delete_f_idx].clear()

        torch.cuda.empty_cache()
        if stop_frame is not None and out_frame_idx >= stop_frame:
            break


def find_entity_to_add(all_tracks, entity, start_frame, fill_coverage_thrd, pix_perc_thrd):
    """ 
    fill_coverage_thrd: For any new entity mask to add, the existing tracks must cover it < fill_coverage_thrd
    pix_perc_thrd: pixel percentage threshold for adding new object
    """ 
    entity_masks_rle = entity[str(start_frame)]
    if len(entity_masks_rle) == 0:
        return []   # No super pixels found, skipped

    all_pred_masks = list(all_tracks[start_frame].values())
    img_h, img_w = next(iter(entity_masks_rle.values()))['size']

    if len(all_pred_masks) == 0:
        pred_cover_rle = bmask_to_rle(np.zeros((img_h, img_w), dtype=np.uint8) > 0)
    else:
        pred_cover_rle = MaskUtils.merge([x if type(x) is dict else bmask_to_rle(x) for x in all_pred_masks], intersect=0)

    coverages = np.array([coverage(den_mask=entity_mask_rle, mask2intersect=pred_cover_rle) for entity_idx, entity_mask_rle in entity_masks_rle.items()])
    pix_perc = np.array([MaskUtils.area(entity_mask_rle) for entity_idx, entity_mask_rle in entity_masks_rle.items()]) / img_h / img_w
    entity_to_add = np.where(np.logical_and(coverages < fill_coverage_thrd, pix_perc > pix_perc_thrd))[0]

    if len(entity_to_add) > 0:
        print('One side IOU:', ' '.join([f'{x:.2f}' for x in coverages[entity_to_add]]), '#Pixel(%):', ' '.join([f'{x:.2f}' for x in 100 * pix_perc[entity_to_add]]))
    return entity_to_add


def find_first_entity_size(entity):
    for frame_entities in entity.values():
        for mask_rle in frame_entities.values():
            return mask_rle['size']
    return None


def next_object_id(tracked_objs):
    return max(tracked_objs.keys()) + 1 if len(tracked_objs) > 0 else 1


def setup(sam2, instance_name, vid_dir, entity_dir):
    entity_path = osp.join(entity_dir, instance_name + '.json')
    print('Loading from', entity_path)
    with open(entity_path, 'r') as f:
        entity = json.load(f)
    sam2.initialize(video_dir=osp.join(vid_dir, instance_name))
    num_frames = sam2.inference_state['num_frames']
    assert num_frames == len(entity.keys())
    return entity, num_frames, sam2


def init_entities(cfg, sam2, entity, all_tracks, tracked_objs, global_confs, start_frame=0):
    init_entity = entity[str(start_frame)]

    sam2.predictor.reset_state(sam2.inference_state)
    all_tracks[start_frame] = dict()
    global_confs[start_frame] = dict()

    if len(init_entity) == 0:
        print(f'No entities found at frame {start_frame}.')
        return

    img_h, img_w = next(iter(init_entity.values()))['size']
    pix_thrd = img_h * img_w * cfg.init_entity_thrd

    num_added = 0
    for mask_idx, mask_rle in init_entity.items():
        if MaskUtils.area(mask_rle) < pix_thrd:
            print(f'Mask {mask_idx} is too small, skipped')
            continue
        add_mask(sam2, int(mask_idx), rle_to_bmask(mask_rle), all_tracks, tracked_objs, global_confs, frame_idx=start_frame)
        num_added += 1

    print(f'Initializing {num_added} objects.')


def remove_object_from_tracks(obj_idx, all_tracks, tracked_objs, global_confs):
    """Remove one object id."""
    tracked_objs.pop(obj_idx, None)
    for _, frame_tracks in all_tracks.items():
        frame_tracks.pop(obj_idx, None)
    for _, frame_confs in global_confs.items():
        frame_confs.pop(obj_idx, None)


def make_prompt_union(prompt_objs):
    """Merge all prompt masks into one union mask."""
    prompt_masks = list(prompt_objs.values())
    assert len(prompt_masks) > 0
    if len(prompt_masks) == 1:
        return prompt_masks[0]
    return MaskUtils.merge(prompt_masks, intersect=0)


def resolve_init_entity_prompt_conflicts(cfg, all_tracks, tracked_objs, global_confs, prompt_union):
    """
    Adapt initial superpixels to all prompt objects.
    """
    ### Remove small objects
    for obj_idx, obj in list(tracked_objs.items()):
        mask = obj['mask']
        mask_perc = MaskUtils.area(mask) / mask['size'][0] / mask['size'][1] if type(mask) is dict else np.mean(mask)
        if mask_perc < cfg.fill_entity_thrd:
            print(f'Object {obj_idx} ({mask_perc*100:.2f}%) is too small, removed.')
            remove_object_from_tracks(obj_idx, all_tracks, tracked_objs, global_confs)

    ### Process overlaps between the prompt and the initial superpixels
    require_updates = dict()
    for obj_idx, obj in list(tracked_objs.items()):
        obj_coverage = float(coverage(den_mask=obj['mask'], mask2intersect=prompt_union))
        if obj_coverage > cfg.fill_coverage_thrd:
            if obj_coverage < cfg.rm_init_entity_thrd:
                print(f'Object {obj_idx} has an overlap of {obj_coverage:.2f} with prompt, modifying.')
                require_updates[obj_idx] = tracked_objs[obj_idx].copy()
            else:
                print(f'Object {obj_idx} has an overlap of {obj_coverage:.2f} with prompt, removed.')

            remove_object_from_tracks(obj_idx, all_tracks, tracked_objs, global_confs)

    for obj_idx, obj in require_updates.items():
        obj_mask_rle = obj['mask']
        obj_mask = rle_to_bmask(obj_mask_rle)
        intersect_mask = rle_to_bmask(MaskUtils.merge([obj_mask_rle, prompt_union], intersect=1))
        obj_mask[intersect_mask] = 0

        new_rle = bmask_to_rle(obj_mask)
        all_tracks[0][int(obj_idx)] = new_rle
        global_confs[0][int(obj_idx)] = float('inf')
        tracked_objs[int(obj_idx)] = {'mask': new_rle, 'init_frame_idx': 0}


def rebuild_tracker_from_frame0(sam2, all_tracks, tracked_objs, global_confs):
    """
    Reset SAM2 and re-prompt every current frame-0 object.
    """
    sam2.predictor.reset_state(sam2.inference_state)
    frame0 = dict(all_tracks.get(0, {}))
    all_tracks[0] = dict()
    global_confs[0] = dict()

    for obj_id in sorted(frame0.keys()):
        if obj_id not in tracked_objs:
            continue
        mask = rle_to_bmask(frame0[obj_id]) if type(frame0[obj_id]) is dict else frame0[obj_id]
        add_mask(sam2, int(obj_id), mask, all_tracks, tracked_objs, global_confs, frame_idx=0)


def add_prompt_objects(sam2, prompt_objs, all_tracks, tracked_objs, global_confs, frame_idx=0):
    """Add all prompt masks as objects in the same SAM2 tracking run."""
    next_obj_id = next_object_id(tracked_objs)
    prompt_id_to_obj_id = dict()

    for prompt_idx, prompt_mask in sorted(prompt_objs.items(), key=lambda x: int(x[0])):
        obj_id = next_obj_id
        add_mask(sam2, obj_id, rle_to_bmask(prompt_mask), all_tracks, tracked_objs, global_confs, frame_idx=frame_idx)
        tracked_objs[obj_id]['prompt_idx'] = prompt_idx
        prompt_id_to_obj_id[prompt_idx] = obj_id
        print(f'Added prompt object {prompt_idx} as Obj {obj_id}.')
        next_obj_id += 1

    return prompt_id_to_obj_id


def compute_save_all_tubes(cfg, sam2, instance_name, entity, all_tracks, tracked_objs, global_confs, final_save_dir, save_prefix=None):
    removed_objects = set()

    propagate(sam2, all_tracks, start_frame=0, removed_objects=removed_objects, global_confs=global_confs, tracked_objs=tracked_objs)

    num_frames = sam2.inference_state['num_frames']
    spacing = cfg.collect_spacing

    for collect_start_frame in range(spacing, num_frames + spacing, spacing):
        collection = set()
        scan_start = max(collect_start_frame - spacing, 1)
        scan_end = min(collect_start_frame, num_frames)

        for start_frame in range(scan_start, scan_end):
            sam2.predictor.reset_state(sam2.inference_state)
            entity_to_add = find_entity_to_add(all_tracks, entity, start_frame, cfg.fill_coverage_thrd, cfg.fill_entity_thrd)
            if len(entity_to_add) == 0:
                continue

            for entity_idx in entity_to_add:
                obj_id = next_object_id(tracked_objs)
                obj_mask_rle = entity[str(start_frame)][str(entity_idx)]
                add_mask(sam2, obj_id, rle_to_bmask(obj_mask_rle), all_tracks, tracked_objs, global_confs, frame_idx=start_frame)
                collection.add(obj_id)

            propagate(sam2, all_tracks, start_frame=start_frame, stop_frame=collect_start_frame, removed_objects=removed_objects, global_confs=global_confs, tracked_objs=tracked_objs)

        if collect_start_frame >= num_frames or len(collection) == 0:
            continue

        print(f'Refining {len(collection)} new objects...')
        sam2.predictor.reset_state(sam2.inference_state)
        for obj_id in collection:
            if obj_id in removed_objects:
                continue
            for reprompt_frame in range(collect_start_frame - 6, collect_start_frame):
                if reprompt_frame in all_tracks and obj_id in all_tracks[reprompt_frame]:
                    sam2.predictor.add_new_mask(
                        inference_state=sam2.inference_state,
                        frame_idx=reprompt_frame,
                        obj_id=obj_id,
                        mask=rle_to_bmask(all_tracks[reprompt_frame][obj_id]),
                    )

        propagate(sam2, all_tracks, start_frame=collect_start_frame, removed_objects=removed_objects, global_confs=global_confs, tracked_objs=tracked_objs)

    print('Saving tubelets...')
    if len(tracked_objs) == 0:
        print('No tracked objects found; nothing to save.')
        return

    dim = find_first_entity_size(entity)
    if dim is None:
        print('No entity masks found in this video; nothing to save.')
        return

    if save_prefix is None:
        save_prefix = instance_name

    for obj_id, obj_data in tracked_objs.items():
        single_obj_tracks = dict()
        for frame_idx, frame_data in all_tracks.items():
            if obj_id in frame_data:
                single_obj_tracks[frame_idx] = {obj_id: frame_data[obj_id]}

        if len(single_obj_tracks) == 0:
            continue

        save_path = osp.join(final_save_dir, f'{save_prefix}_{obj_id}.json')
        save_all_to_json(
            {'all_tracks': single_obj_tracks, 'tracked_objs': {obj_id: obj_data}},
            save_path,
            dim=dim,
        )


def make_config(args, cfg, verbose=True):
    tb_cfg = cfg.tubelet
    final_cfg = SimpleNamespace(
        tracker=tb_cfg.tracker,
        init_entity_thrd=tb_cfg.init_entity_thrd,
        fill_entity_thrd=tb_cfg.fill_entity_thrd,
        fill_coverage_thrd=tb_cfg.fill_coverage_thrd,
        rm_init_entity_thrd=tb_cfg.rm_init_entity_thrd,
        collect_spacing=int(tb_cfg.collect_spacing),
        paths=cfg.paths,
        data=cfg.datasets[args.dataset],
        entity_method=tb_cfg.entity_method,
        entity_dir=osp.join(cfg.paths.intermdir, f'entities_{args.dataset}_{tb_cfg.entity_method}'),
        **args.__dict__,
    )
    if verbose:
        print('Config:')
        for arg in vars(final_cfg):
            print(f'  {arg.replace("_", " ").title():15}: {getattr(final_cfg, arg)}')
    return final_cfg


def get_parser():
    parser = argparse.ArgumentParser(description='Compute all tracks.')
    parser.add_argument('-c', '--config', default='configs/default.yaml', metavar='FILE', help='path to config file')
    parser.add_argument('-d', '--dataset', type=str, default='vost')
    parser.add_argument('-s', '--split', type=str, help='Dataset split to run', default='val')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--wid', type=int, default=0, help='worker id.')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    yml_cfg = load_yaml_file(args.config)
    cfg = make_config(args, yml_cfg)

    with open(osp.join(cfg.data.split_dir, f'{cfg.split}.txt'), 'r') as f:
        instance_names = [x.strip() for x in f.readlines()]
        instance_names.sort()

        if cfg.num_workers > 1:
            random.seed(0)
            random.shuffle(instance_names)
            instance_names = instance_names[cfg.wid::cfg.num_workers]

    final_save_dir = osp.join(cfg.paths.intermdir, f'tubelets_{args.dataset}_{cfg.entity_method}')
    os.makedirs(final_save_dir, exist_ok=True)

    for instance_idx, instance_name in enumerate(instance_names):
        print('=' * 20)
        print(f'Processing ({instance_idx + 1}/{len(instance_names)}) {instance_name}...')

        prompt_objs = dict()
        if hasattr(cfg.data, 'anno_dir') and hasattr(cfg.data, 'anno_format'):
            init_prompt_paths = sorted(glob.glob(osp.join(cfg.data.anno_dir, instance_name, cfg.data.anno_format)))
            if len(init_prompt_paths) > 0:
                prompt_objs = load_anno(init_prompt_paths[0])

        if len(prompt_objs) == 0:
            final_all_tracks_path = osp.join(final_save_dir, f'{instance_name}_0.json')
            if osp.exists(final_all_tracks_path):
                print('Final path found, skipped')
                continue

            sam2 = get_tracker(cfg.tracker)
            entity, num_frames, sam2 = setup(sam2, instance_name, cfg.data.image_dir, cfg.entity_dir)
            start_time = time.time()
            all_tracks = {k: dict() for k in range(num_frames)}
            tracked_objs = dict()
            global_confs = {k: dict() for k in range(num_frames)}

            init_entities(cfg, sam2, entity, all_tracks, tracked_objs, global_confs, start_frame=0)
            compute_save_all_tubes(cfg, sam2, instance_name, entity, all_tracks, tracked_objs, global_confs, final_save_dir)
            save_timing(yml_cfg, f'track_{args.dataset}', instance_name, time.time() - start_time, num_frames)

            sam2.clear_all_cache()
            torch.cuda.empty_cache()

        else:
            final_all_tracks_path = osp.join(final_save_dir, f'{instance_name}_0.json')
            if osp.exists(final_all_tracks_path):
                print('Final path found, skipped')
                continue

            sam2 = get_tracker(cfg.tracker)
            entity, num_frames, sam2 = setup(sam2, instance_name, cfg.data.image_dir, cfg.entity_dir)
            start_time = time.time()
            all_tracks = {k: dict() for k in range(num_frames)}
            tracked_objs = dict()
            global_confs = {k: dict() for k in range(num_frames)}

            init_entities(cfg, sam2, entity, all_tracks, tracked_objs, global_confs, start_frame=0)

            prompt_union = make_prompt_union(prompt_objs)
            resolve_init_entity_prompt_conflicts(cfg, all_tracks, tracked_objs, global_confs, prompt_union)

            rebuild_tracker_from_frame0(sam2, all_tracks, tracked_objs, global_confs)
            prompt_seed_map = add_prompt_objects(sam2, prompt_objs, all_tracks, tracked_objs, global_confs, frame_idx=0)
            with open(osp.join(final_save_dir, f'{instance_name}_prompt_seeds.json'), 'w') as f:
                json.dump({str(k): str(v) for k, v in prompt_seed_map.items()}, f)

            compute_save_all_tubes(cfg, sam2, instance_name, entity, all_tracks, tracked_objs, global_confs, final_save_dir, save_prefix=instance_name)
            save_timing(yml_cfg, f'track_{args.dataset}', instance_name, time.time() - start_time, num_frames)

            sam2.clear_all_cache()
            torch.cuda.empty_cache()

