# ════════════════════════════════════════════════════════════════
# Cell 6: Chạy Semi-Supervised Fine-Tuning + Tự động Test So Sánh
# ════════════════════════════════════════════════════════════════
import os, subprocess, sys
from pathlib import Path

REPO_PATH = Path('/content/dr-diagnostic-system')

# Tự động git pull code mới nhất từ GitHub
if REPO_PATH.exists():
    print("[*] Đang tự động cập nhật code mới nhất từ GitHub...")
    subprocess.run(['git', '-C', str(REPO_PATH), 'fetch', 'origin', 'feat/integrate-grade-model-semi'], check=False)
    subprocess.run(['git', '-C', str(REPO_PATH), 'reset', '--hard', 'origin/feat/integrate-grade-model-semi'], check=False)

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# Xóa cache module của Python
for mod in list(sys.modules.keys()):
    if mod == 'ai' or mod.startswith('ai.'):
        del sys.modules[mod]

from ai.semi_supervised.keras_semi_supervised import train_keras_semi_supervised, load_keras_grade_model, evaluate_on_test_set

if ARTIFACT_PATH.exists():
    raise RuntimeError(f'Artifact đã tồn tại: {ARTIFACT_PATH}. Hãy đổi tên hoặc xóa file cũ để train mới.')

best_model_path = train_keras_semi_supervised(
    model_path=str(CHECKPOINT_PATH),
    labeled_csv=str(LABELED_CSV),
    unlabeled_dir=str(UNLABELED_DIR),
    output_dir=str(OUTPUT_DIR),
    threshold=SEMI_CONFIG['threshold'],
    pseudo_weight=SEMI_CONFIG['pseudo_weight'],
    epochs=SEMI_CONFIG['epochs'],
    batch_size=SEMI_CONFIG['batch_size'],
    lr=SEMI_CONFIG['lr'],
    input_size=(300, 300),
)

if Path(best_model_path).resolve() != ARTIFACT_PATH.resolve():
    raise RuntimeError(f'Artifact được lưu sai đường dẫn: {best_model_path}')

print('\n[v] Checkpoint semi tốt nhất đã lưu tại:', best_model_path)

# ── TỰ ĐỘNG ĐÁNH GIÁ VÀ SO SÁNH VỚI MODEL CŨ ──
print("\n" + "═"*60)
print("[*] Đang tự động đánh giá Mô Hình Mới sau khi Train Bán Giám Sát...")
print("═"*60)
new_model = load_keras_grade_model(str(best_model_path), input_shape=(300, 300, 3))
post_eval_metrics = evaluate_on_test_set(
    model=new_model,
    test_dir=LABELED_ROOT,
    input_size=(300, 300),
    batch_size=16,
)

if 'pre_eval_metrics' in globals() and pre_eval_metrics is not None:
    old_acc = pre_eval_metrics['overall_accuracy']
    new_acc = post_eval_metrics['overall_accuracy']
    diff = new_acc - old_acc
    print(f"\n📊 BẢNG SO SÁNH KẾT QUẢ ACCURACY MÔ HÌNH:")
    print(f"   - Mô hình CŨ  (Base Model) : {old_acc*100:.2f}%")
    print(f"   - Mô hình MỚI (Semi Model) : {new_acc*100:.2f}%")
    print(f"   - Mức độ cải thiện        : {diff*100:+.2f}% {'🚀' if diff >= 0 else '⚠️'}")
