import os.path as osp
import numpy as np
import colorsys
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # add proj dir to path
from utils import rle_to_bmask, apply_anno


GLOBAL_PALETTE = np.array([colorsys.hsv_to_rgb(i / 10, 1, 1) for i in range(10)]) * 255
np.random.RandomState(10).shuffle(GLOBAL_PALETTE)

def get_color_index(oid):
    oid_str = str(oid)
    numeric_id = int(''.join(filter(str.isdigit, oid_str)) or 0)
    if oid_str.startswith('Pair'): offset = 0
    elif oid_str.startswith('F'): offset = 10
    elif oid_str.startswith('R'): offset = 20
    else: offset = 15
    return (offset + numeric_id * 11) % len(GLOBAL_PALETTE)

def build_node_to_obj_id(pairs):
    """ Map each matched tubelet to the pair it belongs to. """
    return {
        str(node): f'Pair_{pair_idx}'
        for pair_idx, pair_nodes in enumerate(pairs)
        for node in pair_nodes
    }


def get_focus_obj_ids(events, node_to_obj_id):
    """ Objects involved in events, which are the ones a render focuses on. A tubelet that was
    matched becomes the pair it belongs to; one that was not keeps its own id.
    """
    return {
        node_to_obj_id.get(str(node), node)
        for event in events
        for node in (event['parent'], event['child'])
    }


def build_obj_colors(final_masks, focus_obj_ids):
    """ Map every object in final_masks to a palette color. Focus objects with masks are numbered
    first, then all others, so every renderer gives the same object the same color.
    """
    first = sorted(str(obj_id) for obj_id in focus_obj_ids if final_masks.get(obj_id))
    rest = sorted(str(obj_id) for obj_id in final_masks if str(obj_id) not in set(first))
    return {obj_id: GLOBAL_PALETTE[i % len(GLOBAL_PALETTE)] for i, obj_id in enumerate(first + rest)}


def overlay(frame, mask_dict, obj_colors=None):
    vis = frame.copy()
    for oid, rle in mask_dict.items():
        if rle is None:
            continue
        mask = rle_to_bmask(rle)
        if obj_colors is not None and str(oid) in obj_colors:
            color = obj_colors[str(oid)]
        else:
            color = GLOBAL_PALETTE[get_color_index(oid)]
        vis = apply_anno(vis, mask=mask, mask_color=color)
    return vis
