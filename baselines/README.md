# Baselines

Run the commands below from the repository root to reproduce and evaluate each baseline.

## VLM + detector

### GPT-4.1 + GDINO

```bash
python3 baselines/vlm_detector/run.py -c configs/default.yaml -d ours -s val --detector gdino --gpus 0 1 2 3
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/gpt-4.1_gdino_16_final
```

### GPT-4.1 + SAM3

```bash
python3 baselines/vlm_detector/run.py -c configs/default.yaml -d ours -s val --detector sam3 --gpus 0 1 2 3
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/gpt-4.1_sam3_16_final
```

## TubeletGraph variants

### Setup (once)

```bash
git clone https://github.com/YihongSun/TubeletGraph thirdparty/TubeletGraph
cd thirdparty/TubeletGraph
git checkout fdb05b6
git apply ../../baselines/tubeletgraph/rev_prompt.patch
cd ../..
```

The patch adds `--rev`, which prompts for assembly instead of decomposition. 

### TubeletGraph

```bash
python3 baselines/tubeletgraph/make_prompts.py -d ours -s val
# in thirdparty/TubeletGraph/configs/default.yaml, under datasets.ours:
#   image_dir: ../../OmniST/JPEGImages
#   anno_dir: ../../_interm_out/tubeletgraph_prompts_ours
cd thirdparty/TubeletGraph
python3 TubeletGraph/run.py -c configs/default.yaml -d ours -s val -m Ours --gpus 0 1 2 3
cd ../..
python3 baselines/tubeletgraph/to_omnist.py \
    --in_dir thirdparty/TubeletGraph/_pred_out/ours-val-Ours_gpt-4.1 \
    --out_dir _pred_out_baselines/TubeletGraph_Baseline_F -d ours -s val
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_F
```

### TubeletGraph-R

Prompts come from the last frame, and events are mapped back to forward time.

```bash
python3 baselines/tubeletgraph/make_prompts.py -d ours -s val --rev
# in thirdparty/TubeletGraph/configs/default.yaml, under datasets.ours:
#   image_dir: ../../OmniST/JPEGImages
#   anno_dir: ../../_interm_out/tubeletgraph_prompts_ours_rev
cd thirdparty/TubeletGraph
python3 TubeletGraph/run.py -c configs/default.yaml -d ours -s val -m Ours_rev --rev --gpus 0 1 2 3
cd ../..
python3 baselines/tubeletgraph/to_omnist.py \
    --in_dir thirdparty/TubeletGraph/_pred_out/ours-val-Ours_rev_gpt-4.1 \
    --out_dir _pred_out_baselines/TubeletGraph_Baseline_R -d ours -s val --rev
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_R
```

### TubeletGraph-Bidir

```bash
python3 baselines/tubeletgraph/merge_bidir.py -d ours -s val
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_B
```

### TubeletGraph-Bidir + Verif. & Realign.

```bash
python3 baselines/tubeletgraph/verify_realign.py -c configs/default.yaml -d ours -s val \
    --in_dir _pred_out_baselines/TubeletGraph_Baseline_B
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/TubeletGraph_Baseline_B_gpt-4.1
```

## FluxGraph

```bash
git clone https://github.com/YihongSun/FluxGraph thirdparty/FluxGraph
cd thirdparty/FluxGraph
git checkout 09f50ef
cd ../..
python3 baselines/tubeletgraph/make_prompts.py -d ours -s val
# in thirdparty/FluxGraph/configs/default.yaml, under datasets.ours:
#   image_dir: ../../OmniST/JPEGImages
#   anno_dir: ../../_interm_out/tubeletgraph_prompts_ours
cd thirdparty/FluxGraph
python3 FluxGraph/run.py -c configs/default.yaml -d ours -s val -m Ours --gpus 0 1 2 3
cd ../..
python3 baselines/tubeletgraph/to_omnist.py \
    --in_dir thirdparty/FluxGraph/_pred_out/ours-val-Ours_gpt-4.1 \
    --out_dir _pred_out_baselines/FluxGraph -d ours -s val
python3 eval/eval_omnist.py -c configs/default.yaml -d ours -s val -p _pred_out_baselines/FluxGraph
```
