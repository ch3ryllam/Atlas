# OmniST Dataset

**OmniST** is available on Hugging Face [here](https://huggingface.co/datasets/ch3ryllam/OmniST).

## Overview
OmniST is a dataset for evaluating object tracking through both decomposition and assembly transformations. It contains:
- **97 videos**
- **352 annotated objects**
- **54,369 object masks**
- **302 transformations** — 158 decomposition and 144 assembly events

## Annotations and Visualizations
After downloading, the dataset is organized as:
```text
OmniST/
├── Annotations/
├── JPEGImages/
├── ImageSets/
│   └── val.txt
└── Videos/
```

- `JPEGImages/` contains the video frames
- `Annotations/` contains the object and transformation annotations used for evaluation
- `ImageSets/val.txt` lists the videos in the evaluation split
- `Videos/` contains the source videos as `.mp4` files at 15 fps


You can explore the dataset interactively using [demo.ipynb](demo.ipynb)!
<!-- ```bash
jupyter notebook demo.ipynb
``` -->

## Dataset Details
Please refer to the [manuscript](https://arxiv.org/pdf/2511.04678) for more details regarding dataset construction and evaluation metrics.

## Evaluation
Please refer to [here](../README.md#omnist) on how to evaluate OmniST.

## Citation
Coming soon!
<!-- If you find OmniST useful in your research, please consider citing our paper:
```bibtex
@article{sun2025tracking,
  title={Tracking and Understanding Object Transformations},
  author={Sun, Yihong and Yang, Xinyu and Sun, Jennifer J and Hariharan, Bharath},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
``` -->
