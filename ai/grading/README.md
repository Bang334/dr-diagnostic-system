# DR grading

This module trains and serves a five-grade ICDR diabetic-retinopathy classifier:

`0=No DR`, `1=Mild NPDR`, `2=Moderate NPDR`, `3=Severe NPDR`, and
`4=Proliferative DR`.

The primary model is RETFound-DINOv2 ViT-L. The legacy Keras EfficientNet-B3
checkpoint remains deployable through the same predictor interface.

## Dataset layout

The merged APTOS, DDR, IDRiD, EyePACS and Messidor dataset is expected as:

```text
split_dataset/
├── train/{0,1,2,3,4}/*.jpg
├── validation/{0,1,2,3,4}/*.jpg
└── test/{0,1,2,3,4}/*.jpg
```

Before sampling or training, the pipeline audits the complete dataset for:

- exact file and decoded-pixel duplicates;
- perceptually similar images;
- known patient IDs appearing in multiple splits;
- conflicting labels inside duplicate clusters;
- source and patient-ID coverage.

Reports are written to `<output-dir>/audit`. The default `--leakage-policy
error` stops training for exact cross-split duplicates, patient overlap or
conflicting duplicate labels. Perceptual matches are reported for manual review.

Run the audit without loading a GPU model:

```bash
python -m ai.grading.train \
  --dataset-dir /content/fundus_merged/split_dataset \
  --output-dir /content/drive/MyDrive/dr_runs/data_audit \
  --audit-only
```

## Recommended baseline

Install the pinned training environment and authenticate with Hugging Face:

```bash
pip install -r ai/grading/requirements-train.txt
huggingface-cli login
```

Train the final six ViT blocks and classifier head:

```bash
python -m ai.grading.train \
  --dataset-dir /content/fundus_merged/split_dataset \
  --output-dir /content/drive/MyDrive/dr_runs/retfound_last6_seed42 \
  --model-source retfound \
  --retfound-id RETFound_dinov2_meh \
  --image-size 224 \
  --adaptation last_n_blocks \
  --last-n-blocks 6 \
  --batch-size 2 \
  --accum-steps 8 \
  --epochs 30 \
  --warmup-epochs 3 \
  --peak-lr 1e-4 \
  --layer-decay 0.75 \
  --min-lr 1e-6 \
  --weight-decay 0.05 \
  --max-train-images-per-grade 700 \
  --max-eval-images-per-grade 0 \
  --loss ce \
  --balance none \
  --seed 42
```

`--max-eval-images-per-grade 0` keeps the complete validation and test splits.
Model selection uses validation QWK; the test split is evaluated after training.

The supported adaptation modes are:

- `linear_probe`: classifier head only;
- `last_n_blocks`: classifier, final normalization and the final N ViT blocks;
- `full_finetune`: all parameters.

For a linear probe, start with `--peak-lr 3e-4 --warmup-epochs 1`. Keep the
same dataset manifests and seed when comparing adaptation modes.

Resume requires the same model, preprocessing, manifests, adaptation, batch,
accumulation and scheduler configuration:

```bash
--resume /content/drive/MyDrive/dr_runs/retfound_last6_seed42/checkpoint-last.pth
```

Legacy checkpoints without versioned metadata can load their model weights, but
their optimizer and scheduler state are intentionally not restored.

## Artifact contract

Each new `.pth` checkpoint contains:

- model framework, architecture, output type and class names;
- image size, crop tolerance, enhancement flag, interpolation, mean and std;
- adaptation and training configuration;
- Git commit, package versions and SHA-256 of all split manifests;
- optimizer, AMP scaler, scheduler, global update and RNG state in
  `checkpoint-last.pth`.

Additional outputs include `artifact_metadata.json`, `history.jsonl`, probability
columns in `test_predictions.csv`, macro AUROC/AUPRC, ECE, multiclass Brier
score, a normalized confusion matrix and a reliability diagram.

## Inference

Select a model with environment variables:

```bash
DR_MODEL_PATH=/models/checkpoint-best.pth
DR_MODEL_BACKEND=auto
uvicorn ai.grading.main:app --host 0.0.0.0 --port 8001
```

`auto` selects the PyTorch adapter for `.pth`/`.pt` and the legacy Keras adapter
for `.keras`/`.h5`. PyTorch preprocessing comes exclusively from checkpoint
metadata. `DR_PREPROCESS_ENHANCE` is retained only for legacy Keras artifacts
that have no embedded preprocessing contract.
