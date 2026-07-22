# ════════════════════════════════════════════════════════════════
# Cell 4: Cấu hình đường dẫn + widget chọn checkpoint
# ════════════════════════════════════════════════════════════════
import ipywidgets as widgets
from IPython.display import display

DRIVE_ROOT = Path('/content/drive/MyDrive')
ARTIFACT_PATH = DRIVE_ROOT / 'best_semi_EfficientNetB3.keras'
CHECKPOINT_NAME = 'best_EfficientNetB3_rgb_crop_v1.keras'
LABELED_KAGGLE_DATASET = 'sehastrajits/fundus-aptosddridirdeyepacsmessidor'
UNLABELED_KAGGLE_DATASET = 'griffchristenson/unlabeled-retinal-image-dataset'

checkpoint_candidates = []
for current_root, _, files in os.walk(DRIVE_ROOT):
    if CHECKPOINT_NAME in files:
        checkpoint_candidates.append(str(Path(current_root) / CHECKPOINT_NAME))
DEFAULT_CHECKPOINT_PATH = str(DRIVE_ROOT / CHECKPOINT_NAME)
checkpoint_candidates = list(dict.fromkeys(sorted(checkpoint_candidates) + [DEFAULT_CHECKPOINT_PATH]))

checkpoint_widget = widgets.Combobox(
    options=checkpoint_candidates,
    value=checkpoint_candidates[0],
    description='Checkpoint:',
    ensure_option=False,
    layout=widgets.Layout(width='95%'),
)
display(checkpoint_widget)
print('Labeled Kaggle  :', LABELED_KAGGLE_DATASET)
print('Unlabeled Kaggle:', UNLABELED_KAGGLE_DATASET)
print('Artifact        :', ARTIFACT_PATH)
