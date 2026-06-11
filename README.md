# Adaptive and Detection-Aware Super-Resolution for Small Object Detection

This repository contains the implementation used for the paper **"Adaptive and Detection-Aware Super-Resolution for Small Object Detection"**.

The project investigates image super-resolution using SRCNN-based models for a scale factor of **x4**. Starting from a classical SRCNN baseline, the implementation explores residual learning, Charbonnier loss, adaptive region-based loss functions, and detection-aware evaluation using YOLOv8 on COCO validation images.

The main goal is to analyze whether improvements in image reconstruction quality also lead to better object detection performance, especially for small objects. The experiments show that better reconstruction metrics such as PSNR and SSIM do not always translate into improved detection results. For this reason, the project also evaluates a detection-aware variant combined with residual scaling, which controls the intensity of the super-resolution correction applied over the bicubic baseline.

---

## Main Contributions

* Implementation and evaluation of an SRCNN-based image super-resolution pipeline for a scale factor of **x4**.
* Development of an adaptive three-region loss based on **smooth**, **texture**, and **edge** areas.
* Evaluation of super-resolution models using reconstruction metrics such as **PSNR**, **SSIM**, **LPIPS**, and **EdgeMAE**.
* Analysis of the impact of super-resolution on object detection using **YOLOv8** and COCO validation images.
* Experimental demonstration that higher reconstruction quality does not automatically improve object detection performance.
* Introduction of a residual scaling mechanism for controlling the strength of the super-resolution correction.
* Evaluation of the detection-aware residual scaling approach on small objects from the COCO validation dataset.

---

## Repository Structure

```txt
adaptive-detection-aware-super-resolution/
│
├── scripts/
│   ├── prepare_coco.py
│   ├── train_srcnn_baseline.py
│   ├── train_3region_thr1035.py
│   ├── train_detection_aware.py
│   └── evaluate_yolo_small_objects.py
│
├── notebooks/
│   ├── 01_compare_srcnn_models.ipynb
│   ├── 02_yolo_sr_detection_compare.ipynb
│   ├── 03_detection_aware_comparison.ipynb
│   └── 04_residual_scaling_alpha015.ipynb
│
├── outputs/
│   └── results
│
├── data/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Method Overview

The project follows a controlled experimental pipeline:

```txt
High-resolution image
        ↓
Bicubic downsampling x4
        ↓
Bicubic upsampling
        ↓
SRCNN-based super-resolution model
        ↓
Reconstructed image
        ↓
YOLOv8 object detection evaluation
```

The implemented models include:

1. **SRCNN baseline**
   Classical SRCNN-based super-resolution model adapted for x4 reconstruction.

2. **SRCNN with adaptive three-region loss**
   The image is divided into smooth, texture, and edge regions using gradient information. Different loss components are applied depending on the region type.

3. **Detection-Aware Super-Resolution**
   A variant designed to preserve structures that are useful for object detection, especially for small objects.

4. **Detection-Aware Super-Resolution with Residual Scaling**
   The final reconstructed image is computed as:

```txt
SR = Bicubic + alpha * Residual
```

where `alpha` controls how much of the learned SRCNN residual is applied over the bicubic image.

---

## Datasets

The datasets are not included in this repository because of their size and licensing restrictions.

The experiments use:

* **DIV2K** for training and validation of the super-resolution models.
* **Set5** for super-resolution testing and visual comparison.
* **Set14** for super-resolution testing and visual comparison.
* **COCO 2017 validation dataset** for YOLO-based object detection evaluation.

Expected local structure:

```txt
data/
├── train_HR/
├── valid_HR/
├── Set5/
├── Set14/
└── coco_da_val/
    ├── images/
    └── instances_val2017.json
```

More details are available in:

```txt
data/README.md
```

---

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```txt
torch
torchvision
numpy
pandas
Pillow
matplotlib
tqdm
ultralytics
```

The YOLOv8 detector is loaded through the `ultralytics` package. The default model used in the experiments is:

```txt
yolov8n.pt
```

Model weights and large datasets are not tracked by Git.

---

## Training

### Train SRCNN baseline

```bash
python scripts/train_srcnn_baseline.py
```

### Train adaptive 3-region SRCNN

```bash
python scripts/train_3region_thr1035.py
```

### Train detection-aware SRCNN

```bash
python scripts/train_detection_aware.py
```

The training scripts generate low-resolution inputs automatically by bicubic downsampling the high-resolution images with a scale factor of x4.

---

## COCO Preparation

To prepare the COCO subset used for detection evaluation:

```bash
python scripts/prepare_coco.py
```

The prepared images and annotations should be placed in:

```txt
data/coco_da_val/
├── images/
└── instances_val2017.json
```

---

## Detection Evaluation

The final YOLO-based evaluation on small COCO objects is implemented in:

```txt
scripts/evaluate_yolo_small_objects.py
```

Run:

```bash
python scripts/evaluate_yolo_small_objects.py
```

This script evaluates the following variants:

```txt
HR
Bicubic
3Region_Thr1035 + alpha = 0.15
DetectionAware_Thr1035_Beta15 + alpha = 0.15
```

The evaluation uses:

```txt
YOLO model: yolov8n.pt
Confidence threshold: 0.10
IoU threshold: 0.50
Object size filter: small
Maximum images: 100
Residual scaling alpha: 0.15
```

Generated tables and examples are saved in:

```txt
outputs/tables/
outputs/examples/
```

---

## Results Summary

### Super-resolution reconstruction results

| Model             | PSNR ↑ | SSIM ↑ | EdgeMAE ↓ | LPIPS ↓ |
| ----------------- | -----: | -----: | --------: | ------: |
| Bicubic           | 25.381 | 0.9571 |    0.0915 |  0.4125 |
| SRCNN V4 baseline | 26.017 | 0.9635 |    0.0834 |  0.3290 |
| Weighted Loss     | 26.045 | 0.9638 |    0.0827 |  0.3290 |
| 3Region Thr1035   | 26.037 | 0.9643 |    0.0805 |  0.3417 |

The adaptive variants obtain moderate improvements over bicubic interpolation and the SRCNN baseline, especially in structural regions such as edges.

### YOLO detection on small COCO objects

| Variant                               | Precision ↑ | Recall ↑ |   F1 ↑ |
| ------------------------------------- | ----------: | -------: | -----: |
| HR                                    |      0.4178 |   0.5619 | 0.4792 |
| Bicubic                               |      0.4630 |   0.3961 | 0.4269 |
| 3Region Thr1035 + alpha = 0.15        |      0.4619 |   0.4028 | 0.4303 |
| DetectionAware Thr1035 + alpha = 0.15 |      0.4675 |   0.4077 | 0.4355 |

The final detection-aware residual scaling variant achieves the best F1 score among the low-resolution reconstruction variants tested on small COCO objects.

---

## Key Observation

The experiments show that super-resolution should not be evaluated only through reconstruction metrics. Although SRCNN-based adaptive models improve PSNR, SSIM, LPIPS, or edge-based metrics, this does not automatically improve object detection performance.

For downstream computer vision tasks, task-aware evaluation is necessary. In this project, the best reconstruction-oriented model is not always the best detection-oriented model. Residual scaling helps reduce aggressive corrections and limits artifacts that may negatively affect the detector.

---

## Notebooks

The notebooks contain exploratory analysis, visual comparisons, and additional evaluation steps:

```txt
notebooks/01_compare_srcnn_models.ipynb
```

Compares bicubic interpolation and SRCNN-based super-resolution variants using reconstruction metrics and visual examples.

```txt
notebooks/02_yolo_sr_detection_compare.ipynb
```

Evaluates the impact of super-resolution preprocessing on YOLO-based object detection.

```txt
notebooks/03_detection_aware_comparison.ipynb
```

Compares the detection-aware variant with the reconstruction-oriented SRCNN variants.

```txt
notebooks/04_residual_scaling_alpha015.ipynb
```

Contains the final residual scaling experiment using alpha = 0.15 on small COCO objects.

---

## Outputs

The `outputs/` folder stores generated results:

```txt
outputs/
├── figures/
├── tables/
└── examples/
```

Recommended output files include:

```txt
outputs/tables/sr_reconstruction_metrics.csv
outputs/tables/yolo_small_objects_residual_scaling_alpha015.csv
outputs/figures/alpha_f1_curve.png
outputs/examples/
```

Large generated outputs should not be committed unless they are small and directly relevant for the paper.

---

## Reproducibility Notes

To reproduce the main experiments:

1. Download and prepare the datasets described in `data/README.md`.
2. Install the dependencies from `requirements.txt`.
3. Train or place the required SRCNN checkpoints locally.
4. Prepare the COCO validation subset.
5. Run the YOLO small-object evaluation script.

The repository focuses on the final reproducible code and paper-relevant experiments. Intermediate experiments, large datasets, and model checkpoints are not included.

---

## Citation

If this repository is used as part of academic work, please cite the related paper:

```txt
C. E. Luncanu, S. Spînu "Adaptive and Detection-Aware Super-Resolution for Small Object Detection," Journal of Military Technology.
```

---

## Author

**Cezica-Elena Luncanu**

Artificial Intelligence for Defense and Security
Military Technical Academy "Ferdinand I"
