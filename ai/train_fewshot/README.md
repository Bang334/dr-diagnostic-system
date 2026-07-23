# Train Few-shot

Pipeline adaptation ít nhãn, nối trực tiếp với checkpoint của `ai/grading`:

```text
ai/grading  ←  ai/train_fewshot
```

Không phụ thuộc `ai/semi_supervised` hoặc `ai/train_semi_v2`.

## Dataset đã chọn: DeepDRiD Regular Fundus

Nguồn chính thức:

- Repository: https://github.com/deepdrdoc/DeepDRiD
- Release cố định: https://github.com/deepdrdoc/DeepDRiD/releases/tag/v1.1
- Bài báo/dataset: https://doi.org/10.1016/j.patter.2022.100512

DeepDRiD phù hợp hơn một Kaggle mirror vì có nguồn chính thức, giấy phép
CC-BY-SA-4.0, năm mức DR, metadata và split challenge rõ ràng. Regular Fundus có
2.000 ảnh từ 500 bệnh nhân và tạo target domain khác với checkpoint grading
nguồn.

Notebook lưu bản tải và dataset đã chuẩn hóa trong
`MyDrive/dr_fewshot/datasets`, nên các lần chạy sau không tải hoặc chuyển đổi lại.

## Logic

1. Nạp `checkpoint-best.pth` của grading, đọc model/preprocessing từ metadata.
2. Chuẩn hóa DeepDRiD v1.1 thành `train/validation/test` theo metadata chính thức.
3. Từ target `train`, chọn đúng `K` ảnh mỗi grade một lần và lưu
   `support_manifest.csv`.
4. Mỗi episode tách tạm query khỏi fixed support; không lấy thêm ảnh target.
5. RETFound encoder sinh embedding; prototype mỗi grade là trung bình embedding
   support của grade đó.
6. Fine-tune projection và block encoder cuối bằng episodic cross-entropy.
7. Chọn checkpoint theo fixed-support episodic loss, không nhìn target test.
8. Mỗi epoch lưu `checkpoint-last.pth`; bản tốt nhất lưu
   `checkpoint-best.pth`. Resume khôi phục optimizer, AMP scaler, epoch,
   patience và RNG của episode sampler.
9. Target test chỉ được đánh giá thủ công bằng `--eval-only` sau khi adaptation.

## Chạy

```powershell
python -m ai.train_fewshot.train `
  --checkpoint D:\runs\grade\checkpoint-best.pth `
  --target-dataset-dir D:\data\deepdrid-prepared `
  --output-dir D:\runs\fewshot-deepdrid-5shot-seed42 `
  --shots 5 `
  --queries 1 `
  --epochs 8
```

Resume:

```powershell
python -m ai.train_fewshot.train `
  ...cùng cấu hình và đường dẫn... `
  --resume D:\runs\fewshot-deepdrid-5shot-seed42\checkpoint-last.pth
```

Test cuối:

```powershell
python -m ai.train_fewshot.train `
  ...cùng cấu hình và đường dẫn... `
  --resume D:\runs\fewshot-deepdrid-5shot-seed42\checkpoint-best.pth `
  --eval-only
```

Artifact:

- `support_manifest.csv`
- `checkpoint-best.pth`
- `checkpoint-last.pth`
- `history.jsonl`
- `training.log`
- `summary.json`
- `comparison.json` sau `--eval-only`
