"""
Run all transformation evaluation metrics for a split.

By default this reads predictions from:
  <cfg.paths.outdir>/<dataset or dataset_ablation>/

and writes results to:
  <cfg.paths.evaldir>/<dataset or dataset_ablation>/

Use -p/--pred to evaluate another method or baseline prediction folder directly.

Usage:
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val --skip_fp
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val --skip_temporal_alignment
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val --skip_fp --skip_temporal_alignment
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/gpt-4.1_gdino_16_final
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_F
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_R
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_B
"""

import argparse
import json
import os
import os.path as osp
import subprocess
import sys

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from utils import load_yaml_file



def run_cmd(cmd, name):
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"\n[!] {name} failed")
        print("Command:", " ".join(cmd))
        print(result.stdout)
        print(result.stderr)
        return False
    return True


def safe_div(num, den, default=0.0):
    return num / den if den > 0 else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str, default="val")
    parser.add_argument("-p", "--pred", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-a", "--anno", type=str, default=None)
    parser.add_argument("-o", "--out_dir", type=str, default=None)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip_fp", action="store_true")
    parser.add_argument("--skip_temporal_alignment", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_file(args.config)
    output_dataset = args.dataset
    if args.skip_fp: output_dataset += "_skip_fp"
    if args.skip_temporal_alignment: output_dataset += "_skip_temporal_alignment"

    data_cfg = getattr(cfg.datasets, args.dataset)
    split_path = osp.join(data_cfg.split_dir, f"{args.split}.txt")
    with open(split_path, "r") as f:
        video_names = [line.strip() for line in f if line.strip()]

    print("\n" + "=" * 60)
    print(f"Starting Evaluation for {len(video_names)} videos.")
    print("=" * 60)

    pred_dir = args.pred or osp.join(cfg.paths.outdir, output_dataset)
    base_evaldir = args.out_dir or cfg.paths.evaldir
    anno_dir = args.anno or data_cfg.anno_dir
    batch_name = f"{args.split}_{osp.basename(osp.normpath(anno_dir))}_{osp.basename(osp.normpath(pred_dir))}_pad{args.pad}"
    out_dir = osp.join(base_evaldir, output_dataset, batch_name)
    json_dir = osp.join(out_dir, "json_stats")
    os.makedirs(json_dir, exist_ok=True)
    all_stats = {
        cat: {"tp": 0, "fp": 0, "fn": 0, "action_num": 0, "action_den": 0, "obj_num": 0, "obj_den": 0,
              "st_pass": 0, "overall_pass": 0, "total_gt": 0, "track_iou_sum": 0.0, "track_gt_count": 0}
        for cat in ["overall", "decomposition", "composition"]
    }
    successful_videos = 0

    for i, video in enumerate(video_names):
        print(f"[{i + 1}/{len(video_names)}] {video} ...", end=" ", flush=True)
        temploc_json = osp.join(json_dir, f"{video}_temploc_stats.json")
        match_json = osp.join(json_dir, f"{video}_matches.json")
        sem_json = osp.join(json_dir, f"{video}_sem_stats.json")
        track_json = osp.join(json_dir, f"{video}_track_stats.json")
        stages = [
            ("Temporal evaluation", ["python3", "eval/metrics/temploc_pr.py", "-c", args.config, "-p", pred_dir, "-a", anno_dir, "-v", video, "--pad", str(args.pad), "--fps", str(data_cfg.fps), "--json_out", temploc_json, "--match_out", match_json]),
            ("Semantic evaluation", ["python3", "eval/metrics/sem_accuracy.py", "-c", args.config, "-p", pred_dir, "-a", anno_dir, "-v", video, "--pad", str(args.pad), "--json_out", sem_json, "--match_in", match_json]),
            ("Tracking evaluation", ["python3", "eval/metrics/track_miou.py", "-i", video, "--anno_dir", anno_dir, "--pred_dir", pred_dir, "--json_out", track_json]),
        ]
        if not all(run_cmd(cmd, name) for name, cmd in stages):
            print(f"[!] Evaluation failed for video: {video}")
            continue

        result = {}
        for key, path in [("temploc", temploc_json), ("semantic", sem_json), ("tracking", track_json)]:
            if osp.exists(path):
                with open(path, "r") as f:
                    result[key] = json.load(f)

        successful_videos += 1
        print("done")
        for cat in ["overall", "decomposition", "composition"]:
            if "temploc" in result and cat in result["temploc"]:
                all_stats[cat]["tp"] += result["temploc"][cat]["tp"]
                all_stats[cat]["fp"] += result["temploc"][cat]["fp"]
                all_stats[cat]["fn"] += result["temploc"][cat]["fn"]

            if "semantic" in result and cat in result["semantic"]:
                for k in ["action_num", "action_den", "obj_num", "obj_den", "st_pass", "overall_pass", "total_gt"]:
                    all_stats[cat][k] += result["semantic"][cat][k]

            all_stats[cat]["track_iou_sum"] += result["tracking"]["stats"][cat]["iou_sum"]
            all_stats[cat]["track_gt_count"] += result["tracking"]["stats"][cat]["gt_count"]

    if successful_videos == 0:
        print("\nNo successful evaluations to summarize.")
        return 1

    split_names = {"overall": "Overall", "decomposition": "Decomposition", "composition": "Composition"}
    count_rows = [" & ".join(["Split", "GT", "TF-Temp TP", "TF-Temp FP", "TF-Temp FN"]) + r" \\"]
    metric_rows = [" & ".join(["Split", "T_P", "T_R", "S_V", "S_O", "H_ST,P", "H_ST", "H_P", "H", "J"]) + r" \\"]
    for cat in ["overall", "decomposition", "composition"]:
        stats = all_stats[cat]
        num_pred = stats["tp"] + stats["fp"]  # precision metrics divide by the number of predictions, as in bootstrap_omnist.py
        temp_pre = safe_div(stats["tp"], num_pred, default=1.0)
        temp_rec = safe_div(stats["tp"], stats["tp"] + stats["fn"], default=1.0)
        sem_verb, sem_obj = safe_div(stats["action_num"], stats["action_den"]), safe_div(stats["obj_num"], stats["obj_den"])
        st_pre = safe_div(stats["st_pass"], num_pred, default=1.0)
        st_rec = safe_div(stats["st_pass"], stats["total_gt"])
        ov_pre = safe_div(stats["overall_pass"], num_pred, default=1.0)
        ov_rec = safe_div(stats["overall_pass"], stats["total_gt"])
        miou = safe_div(stats["track_iou_sum"], stats["track_gt_count"])

        count_values = [split_names[cat], stats["total_gt"], stats["tp"], stats["fp"], stats["fn"]]
        metric_values = [split_names[cat], temp_pre * 100, temp_rec * 100, sem_verb * 100, sem_obj * 100, st_pre * 100, st_rec * 100, ov_pre * 100, ov_rec * 100, miou * 100]
        count_rows.append(" & ".join(str(v) for v in count_values) + r" \\")
        metric_rows.append(" & ".join(str(v) if isinstance(v, str) else f"{v:.1f}" for v in metric_values) + r" \\")
    table_str = "Counts\n" + "\n".join(count_rows) + "\n\nMetrics\n" + "\n".join(metric_rows) + "\n"

    metadata_header = f"Evaluation Summary ({successful_videos}/{len(video_names)} videos successful)\nAnnotations : {anno_dir}\nPredictions : {pred_dir}\nEval Folder : {out_dir}\nPadding     : {args.pad}\n\n"
    print("\n" + table_str)

    with open(osp.join(out_dir, "summary.txt"), "w") as f:
        f.write(metadata_header + table_str)
    with open(osp.join(out_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=4)

    print(f"\nEvaluation complete! Results saved to: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
