# Depth-Estimation-Task
This project is my implementation of a Lightweight Depth Estimation Model from scratch in Pytorch following the architecture provided this [ICLR paper](https://ieeexplore.ieee.org/document/9411998).

It's using the frozen [MobileNetV2](https://github.com/d-li14/mobilenetv2.pytorch) encoder and trains a Decoder composed of [Inverted Residual](https://ieeexplore.ieee.org/document/8578572) Blocks from the MobileNetV2 network. A U-net style architecture is followed. 
### Model Architecture
<img src="images/model_arc.png" width="750">

### Datasets and Training:
- Training was done on an HP-Omen Laptop with RTX 4060 GPU.
- Due to limited compute resources I am using low-resolution version of the KITTI and CityScapes dataset.
- Download the datasets:
    - [KITTI](https://www.kaggle.com/datasets/utsabkaran/kitti-data)
    - [CityScapes](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation)

### Training Loss on KITTI
<img src="images/train_loss.png" width="750">

### Ground Truth vs Predicted Disparity
<img src="images/results.png" width="750">

---

## Repository Structure

```
Depth-Estimation-Task/
├── README.md                          # This file
├── .gitignore                         # Excludes datasets/, logs/, saved_models/
│
├── test_kitti.ipynb                   # Main notebook – trains & evaluates the model on KITTI
├── test_citysc.ipynb                  # Main notebook – trains & evaluates the model on Cityscapes
│
├── scripts/                           # Reusable Python modules
│   ├── __init__.py                    # Package marker
│   └── depth_benchmark.py            # DepthBenchmark class – scale-invariant depth evaluation
│                                      #   metrics (AbsRel, RMSE, SILog, δ-thresholds,
│                                      #   Spearman ρ, ordinal accuracy) with LSQ/minmax alignment
│
├── notebooks/                         # Older / experimental notebooks
│   ├── test.ipynb                     # Early prototype – encoder-decoder depth model on NYU data
│   ├── test_v2.ipynb                  # Refined iteration of test.ipynb with cleaner code
│   ├── test_nyu.ipynb                 # MobileNetV2-based depth model trained on NYU Depth v2
│   └── monocular-depth-estimation-nyuv2.ipynb
│                                      # Kaggle-style notebook using segmentation_models_pytorch
│                                      #   for monocular depth on NYU v2
│
├── images/                            # Figures used in the README
│   ├── model_arc.png                  # Model architecture diagram
│   ├── train_loss.png                 # Training loss curve plot
│   └── results.png                    # GT vs predicted depth comparison
│
└── mobilenetv2/                       # MobileNetV2 encoder (forked submodule)
    ├── imagenet.py                    # Full ImageNet training/eval script (distributed, DALI, etc.)
    ├── LICENSE
    ├── README.md
    ├── models/
    │   └── imagenet/
    │       ├── __init__.py            # Re-exports from mobilenetv2.py
    │       ├── mobilenetv2.py         # Standard MobileNetV2 for ImageNet classification
    │       └── mbnetv2.py             # Modified MobileNetV2 that returns intermediate feature
    │                                  #   maps per stage – used as the encoder backbone
    ├── pretrained/                    # Pre-trained ImageNet weight files (.pth)
    │   ├── mobilenetv2-c5e733a8.pth   # Default weights used by the project
    │   └── ...                        # Various width-multiplier & resolution variants
    └── utils/
        ├── __init__.py                # Re-exports all utility submodules
        ├── dataloaders.py             # ImageNet data loaders (PyTorch & NVIDIA DALI backends)
        ├── eval.py                    # Top-k classification accuracy helper
        ├── logger.py                  # Training metric logger (TSV files + matplotlib plots)
        ├── misc.py                    # Helpers: dataset stats, weight init, AverageMeter, etc.
        └── visualize.py              # Image display utils: un-normalise, heatmap colouring, overlays
```

> **Note:** `datasets/`, `logs/`, and `saved_models/` are gitignored and not tracked.

---

## TO-DOs:
- Make the Dataloading modular(create different Loading classes for each of dataset and mention the format needed)
- create scripts for the classes and functions. Keep Notebooks at just the testbeds.
 - Improve accuracy and segmentation.
 - Compare with benchmark models like MonoDepth2
 - Try which different loss functions to reduce noise and smoothen output.
 - Improve Segmentation and fix the visualisation