# DR grading training

`train.py` fine-tunes a five-grade APTOS classifier and deliberately keeps the
held-out test split untouched until the best validation-QWK checkpoint has been
selected.

## Colab setup

Run from the project root. Copy/unzip the image dataset into `/content` for fast
reads and keep the output directory on Drive so runtime interruptions do not
lose checkpoints.

```bash
git clone https://github.com/Bang334/dr-diagnostic-system.git
cd dr-diagnostic-system
pip install -r ai/grading/requirements-train.txt
```

Access to `YukunZhou/RETFound_dinov2_meh` on Hugging Face must be approved
before the first RETFound run.

Do **not** paste a Hugging Face token into this repository or notebook source.
In Colab, open the key icon (**Secrets**), create a secret named `HF_TOKEN`,
grant notebook access, then run this private setup cell:

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
assert os.environ["HF_TOKEN"], "HF_TOKEN is missing from Colab Secrets"
```

As an alternative in an interactive terminal, run `huggingface-cli login`.
The GitHub repository owner/user is `Bang334`; no Hugging Face or GitHub token
is stored in tracked files.

When Colab Secrets are unavailable (for example, a Colab kernel attached from
VS Code), the notebook falls back to a hidden `getpass` prompt. The supplied
value exists only in the current runtime environment and must be entered again
after a runtime reset.

For the APTOS download cell, create a second Colab secret named
`KAGGLE_API_TOKEN` using a token generated at Kaggle **Settings → API**. The
notebook exports it only to the current runtime. You must also accept the APTOS
competition rules before the Kaggle CLI can download the files.

## Recommended T4 16 GB baseline

```bash
python -m ai.grading.train \
  --images-dir /content/aptos/train_images \
  --labels-csv /content/aptos/train.csv \
  --output-dir /content/drive/MyDrive/dr_runs/retfound_cbce_seed42 \
  --model-source retfound \
  --retfound-id RETFound_dinov2_meh \
  --image-size 224 \
  --batch-size 2 \
  --accum-steps 8 \
  --freeze-epochs 3 \
  --epochs 30 \
  --loss ce \
  --balance effective
```

Resume an interrupted run with:

```bash
python -m ai.grading.train <same arguments> \
  --resume /content/drive/MyDrive/dr_runs/retfound_cbce_seed42/checkpoint-last.pth
```

The split CSV files are created once under `<output-dir>/splits` and reused.
Keep them with every experiment so all backbones are compared on identical
images. For an ordinal ablation, use `--loss coral --balance sampler`; do not
combine CORAL with effective-number class weights in the same experiment.

Outputs include:

- `checkpoint-best.pth`, selected by validation quadratic weighted kappa;
- `checkpoint-last.pth`, suitable for Colab resume;
- `history.jsonl`;
- `test_metrics.json` and `test_predictions.csv`;
- `confusion_matrix_normalized.png`.

## Lightweight fallback

If RETFound does not fit the assigned GPU, keep every other setting fixed and
change only the model:

```bash
python -m ai.grading.train <data/output arguments> \
  --model-source timm \
  --model-name convnext_tiny.fb_in22k_ft_in1k \
  --image-size 384 \
  --batch-size 8 \
  --accum-steps 2 \
  --loss ce \
  --balance effective
```

Do not initialize from a checkpoint already fine-tuned on APTOS when evaluating
on a new APTOS split; that can leak held-out images into training.

## Preprocessing contract

Training and the current Keras API now share the colour-preserving crop in
`ai.preprocessing.fundus_prep`. The default is crop + resize with RGB retained.
If an experiment is trained with `--enhance`, deploy it with
`DR_PREPROCESS_ENHANCE=1`; otherwise leave that variable unset. A PyTorch API
handler for the resulting RETFound checkpoint can be added after the first
validated checkpoint exists.
