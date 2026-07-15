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

## Colab setup

Use the supplied `DR_Training_Colab.ipynb`. It checks out
`feat/limited-grade-sampling`, downloads the Kaggle dataset, stores checkpoints
on Drive, and can resume from `checkpoint-last.pth` after a runtime reset.

Create these private Colab Secrets and grant the notebook access:

- `KAGGLE_API_TOKEN`: generated from Kaggle Settings → API;
- `HF_TOKEN`: a Hugging Face read token with approved access to
  `YukunZhou/RETFound_dinov2_meh`.

Never paste either token into a notebook cell or commit one to Git. When Colab
Secrets are unavailable, the notebook uses a hidden session-only prompt.

The notebook trains on at most 500 images from each grade (2,500 training
images total) so the pipeline can be validated before committing to a full-data
run. Validation and test remain unchanged to preserve meaningful evaluation.
The sample is deterministic for a given `--seed`.

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
  --epochs 30 \
  --max-train-images-per-grade 500 \
  --loss ce \
  --balance none
```

The capped training set is balanced when every grade has at least 500 images,
so the baseline uses `--balance none`. Set
`--max-train-images-per-grade 0` only when intentionally returning to the full
training set.
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
  --max-train-images-per-grade 500 \
  --loss ce \
  --balance none
```

The legacy `--images-dir` plus `--labels-csv` input remains supported for older
APTOS experiments, but it cannot be combined with `--dataset-dir`.

## Preprocessing contract

Training and the current Keras API share the colour-preserving crop in
`ai.preprocessing.fundus_prep`. The default is crop + resize with RGB retained.
If an experiment is trained with `--enhance`, deploy it with
`DR_PREPROCESS_ENHANCE=1`; otherwise leave that variable unset.
