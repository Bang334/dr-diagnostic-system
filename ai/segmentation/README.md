# Retinal lesion segmentation

This module segments three diabetic-retinopathy lesions from a colour fundus image:

- microaneurysm (MA)
- hemorrhage (HE)
- hard exudate (EX)

It is a multilabel pixel-classification problem: one sigmoid output channel is used for
each lesion. Background is represented when all three channels are zero.

## Research choice

The first controlled experiment uses the official IDRiD segmentation split. IDRiD is
small enough for Colab, has pixel masks for exactly the requested lesion types, and
keeps a published 54-image training / 27-image testing protocol. The notebook downloads
the Kaggle mirror `dankok/diabetic-retinopathy-image-dataset`, then discovers only its
IDRiD `Segmentation` directory.

| Dataset | Role in this project | Strength | Limitation |
|---|---|---|---|
| IDRiD | Baseline and architecture comparison | Official pixel masks; exact MA/HE/EX task; compact | Only 81 segmentation images |
| DDR | Recommended external validation / second training stage | More lesion-annotated images and varied acquisition | Requires a separate label adapter and license review |
| FGADR | Later robustness experiment | Fine-grained lesion annotations | Different annotation protocol; access and license must be verified |
| e-ophtha | Optional MA/EX external test | Focused lesion benchmarks | Does not cover the complete three-lesion target |

Do not combine these datasets blindly. Annotation definitions, image resolution and
the meaning of a missing mask differ between datasets. Add one adapter per dataset and
report both within-dataset and external-dataset results.

## Model and pretrained weights

The default is U-Net with a ResNet34 encoder pretrained on ImageNet. The encoder weights
are downloaded automatically by `segmentation-models-pytorch`/PyTorch and cached under
`TORCH_HOME`; there is no model file to upload manually. The segmentation decoder and
three-channel head start from random weights.

The controlled comparison exposes:

- `unet`: simple, stable baseline
- `unetplusplus`: nested decoder and denser skip fusion
- `manet`: attention-based U-Net variant available in the same library

All variants use the same encoder, split, image size and loss so the comparison is fair.
RETFound is not the default segmentation pretrained model because its ViT-L checkpoint
does not directly provide the multiscale feature maps expected by a conventional U-Net
decoder. It should be a separate transformer-decoder experiment, not silently loaded
into this baseline.

## Leakage-safe protocol

The official 54 training images are deterministically divided into 43 train and 11
validation images. The official 27 testing images are never used for optimization,
early stopping, checkpoint selection or threshold calibration. After the best checkpoint
is selected, three lesion thresholds are calibrated on validation data and the test set
is evaluated once.

The default loss is `0.6 * Dice + 0.4 * focal`. Dice optimizes overlap while focal loss
reduces domination by easy background pixels and is especially useful for tiny MA masks.
The default input size is 768 pixels to retain more small-lesion detail.

## Run in Colab

Open `Lesion_Segmentation_Colab.ipynb`, select a GPU runtime and run every cell in order.
The notebook:

1. mounts Drive and checks the Kaggle token stored in Colab Secrets;
2. clones this branch and installs the segmentation requirements;
3. downloads and caches the selected Kaggle ZIP in Drive;
4. validates all 81 IDRiD images and mask counts;
5. trains the selected U-Net variant and saves the run to Drive.

Equivalent command:

```bash
python -m ai.segmentation.train \
  --dataset-dir /content/selected_data/idrid_segmentation \
  --output-dir /content/drive/MyDrive/retfound_segmentation/my_run/unet \
  --architecture unet \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --image-size 768 \
  --epochs 40 \
  --batch-size 2 \
  --accum-steps 2
```

Important artifacts are `checkpoint-best.pth`, `summary.json`, `history.jsonl`, the
three split CSV files, and prediction overlays under `previews/`. Compare macro Dice,
macro IoU and per-lesion recall; accuracy alone is misleading for sparse masks.

## Primary references

- IDRiD data: <https://idrid.grand-challenge.org/Data/>
- Segmentation Models PyTorch: <https://github.com/qubvel-org/segmentation_models.pytorch>
- Kaggle mirror used by the notebook: <https://www.kaggle.com/datasets/dankok/diabetic-retinopathy-image-dataset>
