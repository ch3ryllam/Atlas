"""
Usage: python3 Atlas/get_partitions.py -c configs/default.yaml -d vost -s val
"""

import os
import os.path as osp
import sys
import glob
import json
import shutil
import argparse
import subprocess
from multiprocessing import Process

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import load_yaml_file

def run_command(cmd, description, env=None):
    """Run a command and handle errors (Single Process)."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    if env and 'CUDA_VISIBLE_DEVICES' in env:
        print(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"{'='*60}")

    env = env or os.environ.copy()
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)

def run_worker(cmd, description, wid, gpus):
    """Run a single worker process."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpus[wid])
    run_command(cmd + ["--wid", str(wid)], f"{description} (WID={wid})", env=env)

def run_parallel(base_cmd, description, gpus):
    """Run command with multiple workers in parallel."""
    processes = []
    for wid in range(len(gpus)):
        p = Process(target=run_worker, args=(base_cmd, description, wid, gpus))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Error: Worker process failed with exit code {p.exitcode}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config file path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-s", "--split", required=True, help="Dataset split")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs to use (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--rev-suffix", default="_rev")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    entity_dir = osp.join(cfg.paths.intermdir, f"entities_{args.dataset}_{cfg.tubelet.entity_method}")
    common = ["-c", args.config, "-d", args.dataset, "--num_workers", str(len(args.gpus))]
    tubelet_cmd = ["python3", "Atlas/tubelet/compute_tubelets_sam.py", *common]

    entity_cmd = ["python3", "Atlas/entity_segmentation/cropformer.py", *common]
    run_parallel(entity_cmd + ["-s", args.split], "Computing entities (region proposals)", args.gpus)
    run_parallel(tubelet_cmd + ["-s", args.split], "Computing tubelets [forward]", args.gpus)

    if args.split.endswith(args.rev_suffix):
        return

    with open(osp.join(data_cfg.split_dir, f"{args.split}.txt"), "r") as f:
        instances = [line.strip() for line in f if line.strip()]

    # Build reversed frame folders (symlinks to the original frames) and a reverse split.
    rev_split = f"{args.split}{args.rev_suffix}"
    rev_split_path = osp.join(data_cfg.split_dir, f"{rev_split}.txt")
    rev_dirs = [osp.join(data_cfg.image_dir, f"{instance}{args.rev_suffix}") for instance in instances]
    for instance, rev_dir in zip(instances, rev_dirs):
        input_dir = osp.join(data_cfg.image_dir, instance)
        files = sorted(
            p for ext in ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
            for p in glob.glob(osp.join(input_dir, ext))
        )
        if not files:
            raise FileNotFoundError(f"No image files found in {input_dir}")
        ext = osp.splitext(files[0])[1]
        shutil.rmtree(rev_dir, ignore_errors=True)
        os.makedirs(rev_dir)
        for i, target_file in enumerate(reversed(files)):
            os.symlink(osp.abspath(target_file), osp.join(rev_dir, f"{i:06d}{ext}"))
    with open(rev_split_path, "w") as f:
        f.writelines(f"{osp.basename(rev_dir)}\n" for rev_dir in rev_dirs)

    # CropFormer runs per frame, so the reverse entities are the forward ones reindexed.
    for instance in instances:
        with open(osp.join(entity_dir, f"{instance}.json"), "r") as f:
            fwd = json.load(f)
        rev = {str(i): fwd[str(len(fwd) - 1 - i)] for i in range(len(fwd))}
        with open(osp.join(entity_dir, f"{instance}{args.rev_suffix}.json"), "w") as f:
            json.dump(rev, f)

    try:
        run_parallel(tubelet_cmd + ["-s", rev_split], "Computing tubelets [reverse]", args.gpus)
    finally:
        # The reversed frame folders are only read by the reverse pass above, so delete them
        for rev_dir in rev_dirs:
            shutil.rmtree(rev_dir, ignore_errors=True)
        os.remove(rev_split_path)


if __name__ == "__main__":
    main()