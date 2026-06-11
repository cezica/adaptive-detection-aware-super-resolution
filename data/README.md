# Data

This repository does not include the datasets used for training and evaluation.

The `data/` folder is intentionally kept empty, except for this README file. The datasets must be downloaded separately and placed locally before running the training or evaluation scripts.

## Datasets used

The experiments use the following datasets:

* **DIV2K** – used for training and validation of the SRCNN-based super-resolution models.
* **Set5** – used for super-resolution testing and visual comparison.
* **Set14** – used for super-resolution testing and visual comparison.
* **COCO 2017 validation dataset** – used for YOLO-based object detection evaluation, especially for the small-object experiments.

## Local data structure

After downloading the required datasets, organize the local `data/` folder as follows:

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

## Super-resolution training data

The training scripts expect high-resolution images in:

```txt
data/train_HR/
data/valid_HR/
```

In this project, the high-resolution training and validation images come from the DIV2K dataset.

Low-resolution images are generated automatically during training by bicubic downsampling with a scale factor of x4. The low-resolution images are then upsampled back to the original size and used as input for the SRCNN-based models.

## Super-resolution test data

Set5 and Set14 can be placed in:

```txt
data/Set5/
data/Set14/
```

These datasets are used for qualitative and quantitative comparison between bicubic interpolation and the SRCNN-based super-resolution variants.

## COCO detection evaluation data

The YOLO-based detection evaluation uses images and annotations from the COCO 2017 validation dataset.

Expected structure:

```txt
data/coco_da_val/
├── images/
└── instances_val2017.json
```

The `images/` folder should contain the selected COCO validation images, and `instances_val2017.json` should contain the corresponding COCO annotations.

The script `scripts/prepare_coco.py` can be used to prepare the COCO subset used in the experiments.

## Notes

* The datasets are not tracked by Git.
* The folders above should be created only locally.
* Generated results are saved in the `outputs/` folder.
* Model checkpoints are not stored in this folder.
* Large datasets, generated images, and model weights should not be committed to the repository.
