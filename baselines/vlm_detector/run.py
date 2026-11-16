#!/usr/bin/env python3
"""
Dispatch detect + complete_tracks for every video in a split, distributing jobs across
GPUs in parallel. Produces the "<VLM> + <detector>" rows.

Usage:
python3 baselines/vlm_detector/run.py -c configs/default.yaml
python3 baselines/vlm_detector/run.py -c configs/default.yaml --detector sam3
python3 baselines/vlm_detector/run.py -c configs/default.yaml --detector sam3 --video_native
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from utils import load_yaml_file


def run_commands_for_subdir(subdir: str, gpu_id: int, config: str, dataset: str, detector: str, pred: str, video_native: bool):
    """Run the VLM baseline and then object-track completion for one video on one GPU.

    Stops after the first step if it fails. Returns (subdir, gpu_id) for progress logging.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmds = [
        f"python3 baselines/vlm_detector/detect.py -c {config} -d {dataset} -i {subdir} --detector {detector} --vis --vlm_cache{' --video_native' if video_native else ''}",
        f"python3 baselines/vlm_detector/complete_tracks.py -c {config} -d {dataset} -i {subdir} -p {pred} --tracker {'sam3' if detector == 'sam3' else 'sam2'}",
    ]
    for step, cmd in enumerate(cmds, start=1):
        print(f"[GPU {gpu_id}] Running: {cmd}")
        result = subprocess.run(cmd, shell=True, env=env)
        if result.returncode != 0:
            print(f"[GPU {gpu_id}] ERROR (exit {result.returncode}): {cmd}")
            break  # stop second command if first failed
        print(f"[GPU {gpu_id}] Done: {subdir} - step {step}/{len(cmds)}")

    return subdir, gpu_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default="configs/default.yaml", help="VLM is set by vlm.model_name")
    p.add_argument('-d', '--dataset', default="ours")
    p.add_argument('-s', '--split', default="val", help="split name in the dataset's split_dir")
    p.add_argument('--detector', default="gdino", help="gdino | sam3")
    p.add_argument('--video_native', action='store_true', help="Gemini native video input instead of sampled frames")
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = p.parse_args()

    cfg = load_yaml_file(args.config)
    # Final outputs of complete_obj_tracks.py (16 VLM frames, the vlm_det_baseline.py default, or native video)
    pred = f"{cfg.vlm.model_name}_{args.detector}_{'video' if args.video_native else '16'}"
    output_dir = f"{cfg.paths.outdir}_baselines/{pred}_final"

    data_cfg = getattr(cfg.datasets, args.dataset)
    with open(osp.join(data_cfg.split_dir, f"{args.split}.txt")) as f:
        subdirs = [x.strip() for x in f if x.strip()]
    subdirs = [x for x in subdirs if not osp.exists(osp.join(output_dir, f"{x}_final_masks.json"))]
    print(f"Found {len(subdirs)} subdirectories: {subdirs}\n")

    # Assign GPU round-robin per subdir
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(run_commands_for_subdir, subdir, args.gpus[i % len(args.gpus)], args.config, args.dataset, args.detector, pred, args.video_native)
            for i, subdir in enumerate(subdirs)
        ]
        for future in as_completed(futures):
            subdir, gpu_id = future.result()
            print(f"[GPU {gpu_id}] Finished all steps for: {subdir}\n")


if __name__ == "__main__":
    main()
