# DR grading training

`train.py` fine-tunes a five-grade diabetic-retinopathy classifier on the
[merged Fundus dataset](https://www.kaggle.com/datasets/sehastrajits/fundus-aptosddridirdeyepacsmessidor),
which combines APTOS, DDR, IDRiD, EyePACS and Messidor images. The dataset is
about 10.9 GB and already provides `train`, `validation` and `test` directories
under `split_dataset`; the loader preserves these splits instead of randomly
splitting them again.

Expected layout:

```text
split_dataset/
├── train/{0,1,2,3,4}/*.jpg
├── validation/{0,1,2,3,4}/*.jpg
└── test/{0,1,2,3,4}/*.jpg
```

Grades are `0=No DR`, `1=Mild`, `2=Moderate`, `3=Severe`, and
`4=Proliferative DR`.

The canonical definitions live in `taxonomy.py`. They follow the five-level
ICDR scale and expose the corresponding ETDRS levels (`10`, `20`,
`35/43/47`, `53`, and `61-85`) in every inference response. This is a grading
correspondence, not a claim that one-field fundus inference reproduces a full
ETDRS photographic reading protocol.

## Colab setup

Use the supplied `DR_Training_Colab.ipynb`. It checks out
`feat/merged-dataset-training`, downloads the Kaggle dataset, stores checkpoints
on Drive, and can resume from `checkpoint-last.pth` after a runtime reset.

Create these private Colab Secrets and grant the notebook access:

- `KAGGLE_API_TOKEN`: generated from Kaggle Settings → API;
- `HF_TOKEN`: a Hugging Face read token with approved access to
  `YukunZhou/RETFound_dinov2_meh`.

Never paste either token into a notebook cell or commit one to Git. When Colab
Secrets are unavailable, the notebook uses a hidden session-only prompt.

On a T4, allow roughly 10–30 minutes for the 10.9 GB download and about 1–3 days
for 18 RETFound epochs. Actual time depends on Colab storage and GPU allocation.

## Command-line setup

From the project root:

```bash
pip install -r ai/grading/requirements-train.txt
python -m ai.grading.download_data --output-dir /content/fundus_merged
```

The downloader reads `KAGGLE_API_TOKEN` or `~/.kaggle/kaggle.json` and extracts
the data to `/content/fundus_merged/split_dataset`.

Recommended RETFound baseline:

```bash
python -m ai.grading.train \
  --dataset-dir /content/fundus_merged/split_dataset \
  --output-dir /content/drive/MyDrive/dr_runs/retfound_merged_seed42 \
  --model-source retfound \
  --retfound-id RETFound_dinov2_meh \
  --image-size 224 \
  --batch-size 2 \
  --accum-steps 8 \
  --freeze-epochs 3 \
  --epochs 18 \
  --patience 4 \
  --head-lr 5e-5 \
  --backbone-lr 5e-6 \
  --min-lr 5e-7 \
  --weight-decay 0.05 \
  --loss ce \
  --balance none
```

The merged training set is balanced, so the baseline uses `--balance none`.
Use `--balance sampler` only for an intentional ablation. Resume with the same
arguments plus:

```bash
--resume /content/drive/MyDrive/dr_runs/retfound_merged_seed42/checkpoint-last.pth
```

The scanner writes immutable split manifests to `<output-dir>/splits`. Model
selection uses validation QWK; the test split is evaluated only after training
and selection finish.

Outputs include:

- `checkpoint-best.pth` and resumable `checkpoint-last.pth`;
- `history.jsonl`;
- `test_metrics.json` and `test_predictions.csv`;
- `confusion_matrix_normalized.png`;
- `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`.

## Lightweight fallback

If RETFound does not fit the assigned GPU, keep the data and evaluation setup
fixed and change only the backbone:

```bash
python -m ai.grading.train \
  --dataset-dir /content/fundus_merged/split_dataset \
  --output-dir /content/drive/MyDrive/dr_runs/convnext_merged_seed42 \
  --model-source timm \
  --model-name convnext_tiny.fb_in22k_ft_in1k \
  --image-size 384 \
  --batch-size 8 \
  --accum-steps 2 \
  --loss ce \
  --balance none
```

The legacy `--images-dir` plus `--labels-csv` input remains supported for older
APTOS experiments, but it cannot be combined with `--dataset-dir`.

## Required architecture comparison

The auditable presets are EfficientNet-B3, ResNet-50 and ConvNeXt-Tiny. Run all
three with the same data split, seed and preprocessing, then generate
`backbone_comparison.csv`:

```bash
python -m ai.grading.benchmark_backbones \
  --dataset-dir /content/fundus_merged/split_dataset \
  --output-dir /content/drive/MyDrive/dr_runs/backbone_benchmark \
  --preprocessing rgb_crop \
  --seed 42
```

Each run is selected by validation QWK and evaluated once on the held-out test
split. Do not claim that one architecture is superior until all three result
files were produced by this controlled command.

## Preprocessing contract

`preprocessing.py` owns four deterministic recipes: colour-preserving fundus
crop (`rgb_crop`), green-channel extraction (`green`), green-channel CLAHE
(`clahe`) and Ben Graham enhancement (`ben_graham`). Select one with
`--preprocessing`; its complete specification is stored in the checkpoint and
reused automatically for test and PyTorch deployment.

If the primary fundus-field detector cannot find a usable bright region, the
pipeline falls back to the original tolerance-based crop (or the full decoded
image for a completely dark frame) instead of aborting a DataLoader worker.
Training prints `[preprocessing-warning] image=<path>` so questionable source
images remain auditable and can be removed in a later data-quality pass.

`predictor.py` is the single inference interface. `load_predictor()` selects a
Keras adapter for `.keras`/`.h5` or a PyTorch adapter for `.pth`/`.pt`; callers
always receive the same five-grade ICDR/ETDRS response contract.

For a legacy Keras checkpoint, optional metadata can be placed beside the model
as `<model-name>.keras.json`, for example:

```json
{"preprocessing": {"recipe": "ben_graham", "image_size": 300}}
```

PyTorch checkpoints need no sidecar because training writes the grading
contract, class order and full preprocessing specification into the checkpoint.
