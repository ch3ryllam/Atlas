"""
Usage: python quick_run.py --input_dir assets/example/crackers --fps 15
"""

import os.path as osp
import sys
import argparse
import subprocess

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from eval.build_custom_dataset import create_custom_dataset

def run_command(cmd, description, check=True):
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"\n{description} completed successfully\n")
    return result.returncode == 0

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml", help="Path to base config file (default: configs/default.yaml)")
    parser.add_argument("--input_dir", required=True, help="Path to directory containing video frames (required)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--prompt_mask", default=None, help="Optional indexed PNG prompt mask for the first frame")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs to use (e.g., --gpus 0 1 2 3)")
    return parser

def main():
    args = get_parser().parse_args()
    
    print("="*80)
    print("Running Atlas")
    print("="*80)

    if args.prompt_mask and not osp.isfile(args.prompt_mask):
        print(f"Error: prompt mask not found: {args.prompt_mask}")
        sys.exit(1)

    cfg_path_fwd, video_name_fwd = create_custom_dataset(
        args.config,
        args.input_dir,
        args.prompt_mask,
        fps=args.fps,
    )

    gpu_args = [str(g) for g in args.gpus]

    cmd_run = [
        "python3", "Atlas/get_partitions.py",
        "-c", cfg_path_fwd,
        "-d", "custom",
        "-s", video_name_fwd,
        "--gpus",
    ] + gpu_args

    if not run_command(cmd_run, "Compute Fwd/Rev Partitions"):
        sys.exit(1)

    cmd_vis = [
        "python3", "Atlas/get_prediction.py",
        "-c", cfg_path_fwd,
        "-d", "custom",
        "-s", video_name_fwd,
    ]

    if not run_command(cmd_vis, "Get Predictions and Visualizations"):
        sys.exit(1)


if __name__ == "__main__":
    main()
