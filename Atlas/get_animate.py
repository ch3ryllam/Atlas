"""
Usage:
python3 Atlas/get_animate.py -i puzzle
python3 Atlas/get_animate.py -i crackers
python3 Atlas/get_animate.py -i i7rKCchfD2s
python3 Atlas/get_animate.py -c configs/default.yaml -d ours -i crackers
python3 Atlas/get_animate.py -c configs/default.yaml -d ours -i crackers --all_objects
"""

import os, sys, argparse, subprocess

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default='configs/default.yaml', help="Path to config file")
    parser.add_argument("-d", "--dataset", default='ours', help="Dataset name")
    parser.add_argument("-i", "--instance", required=True, help="Instance name")
    parser.add_argument("--skip_fp", action="store_true")
    parser.add_argument("--skip_temporal_alignment", action="store_true")
    parser.add_argument("--all_objects", action="store_true", help="draw every object in the timeline, not just the ones in focus")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    core_args = [
        "-c",
        args.config,
        "-d",
        args.dataset,
        "-i",
        args.instance,
    ]

    if args.skip_fp:
        core_args.append("--skip_fp")
    if args.skip_temporal_alignment:
        core_args.append("--skip_temporal_alignment")

    # both renderers show only the objects involved in events unless --all_objects is passed
    all_objects_arg = ["--all_objects"] if args.all_objects else []

    video_cmd = ["python3", "Atlas/vis/render_video.py"] + core_args + all_objects_arg
    run_command(video_cmd, "Rendering Video")

    pdf_cmd = ["python3", "Atlas/vis/render_pdf.py"] + core_args + all_objects_arg
    run_command(pdf_cmd, "Drawing Timeline")

    animated_cmd = ["python3", "Atlas/vis/render_animated.py"] + core_args
    run_command(animated_cmd, "Rendering Animated Timeline")

    print("\nVisualization Pipeline Complete.")


if __name__ == "__main__":
    main()
