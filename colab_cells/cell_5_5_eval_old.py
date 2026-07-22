# ════════════════════════════════════════════════════════════════
# Cell 5.5: [TÙY CHỌN] Test/Đánh giá Model Cũ (Before Training)
# ════════════════════════════════════════════════════════════════
import os
from pathlib import Path
from ai.semi_supervised.keras_semi_supervised import load_keras_grade_model, evaluate_on_test_set

# Đặt True nếu muốn bỏ qua bước test model cũ để tiết kiệm thời gian
SKIP_PRE_EVAL = False

pre_eval_metrics = None
if not SKIP_PRE_EVAL:
    print("[*] Đang nạp model cũ để đánh giá trên tập dữ liệu có nhãn...")
    old_model = load_keras_grade_model(str(CHECKPOINT_PATH), input_shape=(300, 300, 3))
    
    # Đánh giá trên tập labeled dataset (LABELED_ROOT)
    pre_eval_metrics = evaluate_on_test_set(
        model=old_model,
        test_dir=LABELED_ROOT,
        input_size=(300, 300),
        batch_size=16,
    )
    del old_model
else:
    print("[!] Đã bỏ qua bước đánh giá model cũ.")
