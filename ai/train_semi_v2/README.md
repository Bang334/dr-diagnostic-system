# Train Semi V2

## Ngưỡng riêng cho từng grade

Pipeline pseudo-label tĩnh hỗ trợ một confidence threshold riêng cho mỗi grade:

```powershell
python -m ai.train_semi_v2.train `
  --checkpoint D:\runs\grade\checkpoint-best.pth `
  --dataset-dir D:\data\fundus_merged `
  --unlabeled-dir D:\data\fundus_unlabeled `
  --output-dir D:\runs\semi-class-thresholds `
  --grade-thresholds 0.99,0.90,0.95,0.90,0.93
```

Thứ tự luôn là grade `0,1,2,3,4`. Nếu bỏ `--grade-thresholds`, cả năm grade
tiếp tục dùng `--threshold` (mặc định `0.95`) để tương thích với lệnh cũ.
Danh sách ngưỡng là một phần của cache fingerprint, vì vậy thay một ngưỡng sẽ
tạo lại pseudo-label. Không đổi ngưỡng khi resume; hãy dùng output/run mới.

Pipeline semi-supervised v2 nối trực tiếp với checkpoint do
`ai/grading/train.py` tạo. Model, preprocessing, image size, loss và grading
contract được dựng lại từ metadata của checkpoint grade; không khai báo lại
kiến trúc bằng tay.

Dependency chỉ đi một chiều:

```text
ai/grading  ←  ai/train_semi_v2
```

`train_semi_v2` không import hoặc gọi `ai/semi_supervised` cũ. Module v2 tự sở
hữu dataset adapter, checkpoint/resume, pseudo-label cache và vòng lặp train.

## Logic training

1. Nạp `checkpoint-best.pth` của grading làm teacher và trọng số khởi tạo student.
2. Đọc model source, model name/architecture, preprocessing, image size và loss
   từ metadata checkpoint; không yêu cầu chạy lại cell train grade.
3. Đọc đúng split cố định: `train` làm labeled replay, `validation` để early
   stopping/chọn checkpoint, `test` được giữ kín.
4. Kiểm tra nguồn ảnh unlabeled nằm ngoài toàn bộ train/validation/test, kể cả
   ảnh copy có nội dung byte giống nhau.
5. Tạo fingerprint từ teacher checkpoint, danh sách ảnh, preprocessing, image
   size, threshold và giới hạn mỗi lớp.
6. Nếu fingerprint đã có cache thì đọc pseudo-label; nếu chưa có thì teacher
   dự đoán một lần, lọc confidence theo threshold và lưu cache nguyên tử.
7. Student bắt đầu từ teacher, học trên hợp:
   - ảnh labeled replay với trọng số `1.0`;
   - ảnh pseudo-label với trọng số `pseudo_weight × confidence`.
8. Sau mỗi epoch, đánh giá validation bằng accuracy, macro-F1, balanced
   accuracy, MAE và QWK; QWK quyết định `checkpoint-best.pth`.
9. `checkpoint-last.pth` lưu thêm optimizer, scheduler, AMP scaler, epoch và
   patience để resume chính xác sau khi Colab ngắt.
10. Held-out test chỉ chạy bằng cell/lệnh `--eval-only` sau khi model đã được
    chọn; metric test không ảnh hưởng training.

## Interface

```powershell
python -m ai.train_semi_v2.train `
  --checkpoint D:\runs\grade\checkpoint-best.pth `
  --dataset-dir D:\data\fundus_merged `
  --unlabeled-dir D:\data\fundus_unlabeled `
  --output-dir D:\runs\semi-v2-run-01 `
  --threshold 0.95
```

Các khả năng chính:

- labeled replay từ split `train`, validation chọn model, test không tham gia train;
- `checkpoint-best.pth` và `checkpoint-last.pth`;
- resume đầy đủ model, optimizer, scheduler, scaler, epoch và patience;
- `training.log`, `history.jsonl`, `baseline_metrics.json`, `summary.json`;
- cache pseudo-label dùng chung giữa nhiều run;
- cache được tái sử dụng khi teacher, tập ảnh, preprocessing và threshold giống nhau;
- đổi threshold, teacher checkpoint, tập ảnh hoặc preprocessing sẽ tạo cache mới;
- xóa cặp file cache `.csv`/`.json` sẽ buộc dự đoán lại;
- Ctrl+C giữ checkpoint của epoch hoàn tất gần nhất.

Mặc định cache nằm cạnh output:

```text
runs/
├── pseudo-label-cache-v2/
│   ├── pseudo-labels-<fingerprint>.csv
│   └── pseudo-labels-<fingerprint>.json
└── semi-v2-run-01/
    ├── checkpoint-best.pth
    ├── checkpoint-last.pth
    ├── pseudo_labels.csv
    ├── training.log
    ├── history.jsonl
    └── summary.json
```

Có thể chỉ định cache cố định trên Google Drive:

```powershell
--pseudo-cache-dir /content/drive/MyDrive/dr_semi_v2/pseudo-cache
```

Resume:

```powershell
python -m ai.train_semi_v2.train `
  ...các đường dẫn như run cũ... `
  --resume D:\runs\semi-v2-run-01\checkpoint-last.pth
```

Đánh giá test sau khi chọn model:

```powershell
python -m ai.train_semi_v2.train `
  ...các đường dẫn như run cũ... `
  --resume D:\runs\semi-v2-run-01\checkpoint-best.pth `
  --eval-only
```

Không copy checkpoint semi v2 vào production trước khi đã đánh giá held-out test,
kiểm định nhiều seed/domain và rà soát lâm sàng phù hợp.
