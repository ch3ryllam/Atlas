# Tracking Objects Through Assembly and Decomposition
### [Project Page](https://atlas-track-objects.github.io/) | [Paper](https://arxiv.org/abs/2511.04678) | [Data](https://huggingface.co/datasets/ch3ryllam/OmniST)

Official PyTorch implementation for the paper: "Tracking Objects Through Assembly and Decomposition".

<a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>  

<!-- ![](assets/teaser.png) -->

## ⚙️ Installation
The code is tested with `python=3.10`, `torch==2.7.0+cu126` and `torchvision==0.22.0+cu126` on a RTX A6000 GPU.
```bash
# Clone and setup environment
git clone --recurse-submodules https://github.com/ch3ryllam/Atlas/
cd Atlas/
conda create -n atlas python=3.10 -y
conda activate atlas
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash thirdparty/setup_ckpts.sh

# Install SAM2 with multi-mask predictions
cd thirdparty/sam2
pip install -e .
pip install -e ".[notebooks]"
python setup.py build_ext --inplace
cd ../..

# Install CropFormer
cd thirdparty
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 --no-build-isolation
ln -s "$(pwd)"/Entity/Entityv2/CropFormer detectron2/projects/CropFormer
cd detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops
bash make.sh
cd ../../../../../../../..
# conda install -c conda-forge libstdcxx-ng # if libstdc++ version mismatch occurs
```
In addition, please configure your OpenAI API key (required for GPT-4.1 querying) as follows (add to `~/.bashrc` to persist across sessions).
```bash
export OPENAI_API_KEY="sk-..."
```


## 🔮 Quick Start

**Option 1: Quick Run (Recommended for Single Videos)**
```bash
python quick_run.py \
    --input_dir <VIDEO_FRAME_DIR> \
    [--fps 30]
```
This automatically handles dataset setup and config generation, and generates a video visualization (.mp4) and state graph diagram (.pdf).
- `--input_dir`: Directory containing video frames as individual images (e.g., `0001.jpg, 0002.jpg, ...`)
- `--fps` (optional): Frames per second, default=30

**Example: crackers**
```bash
python quick_run.py --input_dir assets/example/crackers --fps 15
```
The outputs are found under `_pred_out/custom/` and `_vis_out/custom/`.
- To ensure consistency, the expected outputs are pre-computed and found under `./assets/expected_output/`

**Option 2: Manual Config (For Dataset-Wide Evaluation)**

🔹 To run on entire datasets with multiple videos, please add the following lines to `configs/default.yaml` under `datasets`:
```
datasets:
  <data_name>:
    name: <data_name>
    data_dir: <DATA_PATH>
    image_dir: <DATA_PATH>/JPEGImages
    anno_dir: <DATA_PATH>/Annotations
    split_dir: <DATA_SPLIT_PATH>
    image_format: <IMAGE_FORMAT>  # e.g., "*.jpg"
    anno_format: <ANNO_FORMAT>  # e.g., "*.png"
    fps: <FPS>  # for visualization fps
```
🔹 Then, run the following lines to compute dataset-wise predictions.
```bash
python Atlas/get_partitions.py -c configs/default.yaml -d <data_name> -s <split> --gpus 0 1 2 3
python Atlas/get_prediction.py -c configs/default.yaml -d <data_name> -s <split>
```
- Note that `<DATA_SPLIT_PATH>/<split>.txt` should be found.


## 📊 Evaluations
### OmniST

Please first download [OmniST](https://huggingface.co/datasets/ch3ryllam/OmniST/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines.
```bash
python Atlas/get_partitions.py -c configs/default.yaml -d ours -s val --gpus 0 1 2 3
python Atlas/get_prediction.py -c configs/default.yaml -d ours -s val
```

🔹 To evaluate performance, please run the following line.
```bash
python eval/eval_omnist.py -c configs/default.yaml -d ours -s val
```

### VOST

Please first download [VOST](https://www.vostdataset.org/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```bash
python Atlas/get_partitions.py -c configs/default.yaml -d vost -s val --gpus 0 1 2 3
python Atlas/get_prediction.py -c configs/default.yaml -d vost -s val
```

🔹 To evaluate tracking performances, please run the following line.
```bash
python eval/eval_tracking.py -c configs/default.yaml -d vost -s val
```
<!--
| Data-Split-Method | $J$ | $J^S$ | $J^M$ | $J^L$ | $P$ | $R$ | $J_{tr}$ | $J_{tr}^S$ | $J_{tr}^M$ | $J_{tr}^L$ | $P_{tr}$ | $R_{tr}$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vost-val-Ours| 50.9 | 41.3 | 53.0 | 68.6 | 68.1 | 63.7 | 36.7 | 23.6 | 40.2 | 60.1 | 55.2 | 47.0 |

| Data-Split-Method_VLM | Sem-Acc Verb $S_V$ | Sem-Acc Obj $S_O$ | Temp-Loc Pre $T_P$ | Temp-Loc Rec $T_R$ | Overall-Rec(ST) $H_{ST}$ | Overall-Rec $H$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vost-val-Ours_gpt-4.1| 81.8 (*) | 72.3 (*) | 43.1 | 20.4 | 12.0 | 6.5 (*) |

(*) Minor variance may be observed across runs due to non-deterministic LLM behavior in metric computation.
-->


### VSCOS

Please first download [VSCOS](https://venom12138.github.io/VSCOS.github.io/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```bash
python Atlas/get_partitions.py -c configs/default.yaml -d vscos -s val --gpus 0 1 2 3
python Atlas/get_prediction.py -c configs/default.yaml -d vscos -s val
```

🔹 To evaluate tracking performances, please run the following line.
```bash
python eval/eval_tracking.py -c configs/default.yaml -d vscos -s val
```
<!--
| Data-Split-Method | $J$ | $J^S$ | $J^M$ | $J^L$ | $P$ | $R$ | $J_{tr}$ | $J_{tr}^S$ | $J_{tr}^M$ | $J_{tr}^L$ | $P_{tr}$ | $R_{tr}$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|vscos-val-Ours| 75.9 | 67.8 | 79.1 | 81.0 | 89.3 | 82.9 | 72.2 | 60.7 | 77.6 | 78.4 | 87.4 | 81.7 |
-->


### M<sup>3</sup>-VOS

Please first download [M<sup>3</sup>-VOS](https://zixuan-chen.github.io/M-cube-VOS.github.io/) and update the corresponding paths in [configs/default.yaml](configs/default.yaml).

🔹 To compute dataset-wise predictions, please run the following lines. 
```bash
python Atlas/get_partitions.py -c configs/default.yaml -d m3vos -s val --gpus 0 1 2 3
python Atlas/get_prediction.py -c configs/default.yaml -d m3vos -s val
```

🔹 To evaluate tracking performances, please run the following line.
```bash
python eval/eval_tracking.py -c configs/default.yaml -d m3vos -s val
```
<!--
| Data-Split-Method | $J$ | $J^S$ | $J^M$ | $J^L$ | $P$ | $R$ | $J_{tr}$ | $J_{tr}^S$ | $J_{tr}^M$ | $J_{tr}^L$ | $P_{tr}$ | $R_{tr}$ |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|m3vos-val-Ours| 74.1 | 67.4 | 78.7 | 78.2 | 88.4 | 79.8 | 64.1 | 55.9 | 68.5 | 70.3 | 82.4 | 71.5|
-->

## 🖼️ Visualizations

🔹 To visualize model predictions, please run the following line.
```bash
python Atlas/get_animate.py -c <CONFIG> -d <DATASET> -i <INSTANCE>
## example
python Atlas/get_animate.py -c configs/default.yaml -d ours -i crackers
```
- Output visualizations can be found under `_vis_out/<DATASET>/`.

🔹 By default only the objects involved in a transformation are shown. Pass `--all_objects` to show every object instead.
```bash
python Atlas/get_animate.py -c configs/default.yaml -d ours -i crackers --all_objects
```

🔹 To visualize the internally-computed spatiotemporal partition, please run the following line.
```bash
python Atlas/vis/render_partitions.py -c <CONFIG> -d <DATASET> -i <INSTANCE>
## example
python Atlas/vis/render_partitions.py -c configs/default.yaml -d ours -i crackers
```
- Output visualization can be found at `_vis_out/<DATASET>/<INSTANCE>_partitions.mp4`, showing all forward tubelets (top-left), all reverse tubelets (top-right), emergent forward tubelets (bottom-left), and emergent reverse tubelets (bottom-right).


## Citation
Coming soon!
<!-- If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{sun2025tracking,
  title={Tracking and Understanding Object Transformations},
  author={Sun, Yihong and Yang, Xinyu and Sun, Jennifer J and Hariharan, Bharath},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
``` -->
