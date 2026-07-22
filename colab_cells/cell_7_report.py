# ════════════════════════════════════════════════════════════════
# Cell 7: Báo cáo kết quả chi tiết & Lịch sử huấn luyện
# ════════════════════════════════════════════════════════════════
import json
import pandas as pd
from IPython.display import display

pseudo_path = OUTPUT_DIR / 'pseudo_labels.csv'
history_path = OUTPUT_DIR / 'history.json'
for artifact in (pseudo_path, history_path, ARTIFACT_PATH):
    if not artifact.exists():
        raise FileNotFoundError(f'Thiếu artifact: {artifact}')

pseudo_df = pd.read_csv(pseudo_path)
history = json.loads(history_path.read_text(encoding='utf-8'))
print(f'Pseudo-label được giữ: {len(pseudo_df):,}')
if not pseudo_df.empty:
    display(pseudo_df.groupby('pseudo_label').agg(images=('image_path', 'size'), mean_confidence=('confidence', 'mean')))
    display(pseudo_df.head(20))
print(json.dumps(history, indent=2, ensure_ascii=False))
