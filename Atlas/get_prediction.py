"""
Usage:
python3 Atlas/get_prediction.py -c configs/default.yaml -d ours -s val
python3 Atlas/get_prediction.py -c configs/default.yaml -d ours -i crackers
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import load_yaml_file


def run_command(cmd, description):
    print("\n" + "=" * 60)
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with return code {result.returncode}")


def run_for_instance(args, instance):
    core_args = [
        '-c', args.config,
        '-d', args.dataset,
        '-i', instance,
    ]

    ablation_args = []
    if args.skip_fp:
        ablation_args.append('--skip_fp')
    if args.skip_temporal_alignment:
        ablation_args.append('--skip_temporal_alignment')

    detect_cmd = ['python3', 'Atlas/prediction/detect_events.py'] + core_args + ablation_args
    run_command(detect_cmd, f'Detecting Transformations ({instance})')

    vlm_cmd = ['python3', 'Atlas/prediction/prompt_vlm.py'] + core_args + ablation_args
    run_command(vlm_cmd, f'Prompting VLM ({instance})')

    masks_cmd = ['python3', 'Atlas/prediction/build_masks.py'] + core_args + ablation_args
    run_command(masks_cmd, f'Building Final Masks ({instance})')

    animation_cmd = ['python3', 'Atlas/get_animate.py'] + core_args + ablation_args
    run_command(animation_cmd, f'Drawing Animation ({instance})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-d', '--dataset', default='ours')
    parser.add_argument('-s', '--split')
    parser.add_argument('-i', '--instance')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of instances to process in parallel.')
    parser.add_argument('--skip_fp', action='store_true', help='Ablation: disable VLM to filter false positives.')
    parser.add_argument('--skip_temporal_alignment', action='store_true', help='Ablation: disable VLM for temporal alignment.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    cfg = load_yaml_file(args.config)
    if args.instance:
        instances = [args.instance]
    else:
        if not args.split:
            parser.error('one of -i/--instance or -s/--split is required')
        data_cfg = getattr(cfg.datasets, args.dataset)
        split_path = osp.join(data_cfg.split_dir, f'{args.split}.txt')
        with open(split_path, 'r') as f:
            instances = [line.strip() for line in f if line.strip()]

    if args.num_workers < 1:
        parser.error('--num_workers must be >= 1')

    failed = []
    jobs = list(enumerate(instances, start=1))

    if args.num_workers == 1 or len(instances) == 1:
        for idx, instance in jobs:
            print(f'[{idx}/{len(instances)}] Processing {instance}')
            try:
                run_for_instance(args, instance)
            except Exception as exc:
                failed.append((instance, exc))
                print(f'[{idx}/{len(instances)}] Failed {instance}: {exc}')
            else:
                print(f'[{idx}/{len(instances)}] Finished {instance}')
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(run_for_instance, args, instance): (idx, instance)
                for idx, instance in jobs
            }
            for future in as_completed(futures):
                idx, instance = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((instance, exc))
                    print(f'[{idx}/{len(instances)}] Failed {instance}: {exc}')
                else:
                    print(f'[{idx}/{len(instances)}] Finished {instance}')

    if failed:
        print('Failed instances:')
        for instance, exc in failed:
            print(f'  {instance}: {exc}')
        sys.exit(1)

    print(f'Finished all {len(instances)} instances.')


if __name__ == '__main__':
    main()
