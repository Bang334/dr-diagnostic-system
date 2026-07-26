# Train Few-shot

Pipeline adaptation ít nhãn, nối trực tiếp với checkpoint của `ai/grading`:

```text
ai/grading  ←  ai/train_fewshot
```

Không phụ thuộc `ai/semi_supervised` hoặc `ai/train_semi_v2`.

## Dataset đã chọn: DeepDRiD Regular Fundus

Nguồn chính thức:

- Repository: https://github.com/deepdrdoc/DeepDRiD
- Release cố định: https://zenodo.org/records/8248825
- Bài báo/dataset: https://doi.org/10.1016/j.patter.2022.100512

DeepDRiD phù hợp hơn một Kaggle mirror vì có nguồn chính thức, giấy phép
CC-BY-SA-4.0, năm mức DR, metadata và split challenge rõ ràng. Regular Fundus có
2.000 ảnh từ 500 bệnh nhân và tạo target domain khác với checkpoint grading
nguồn.

Notebook tải và chuẩn hóa dataset trong runtime Colab tại `/content/dr_raw` và
`/content/dr`. Dataset không được ghi vào Google Drive.

## Logic

1. Nạp checkpoint grading, đọc model/preprocessing từ metadata.
2. Chuẩn hóa DeepDRiD v1.1 thành `train/validation/test` theo metadata chính thức.
3. Từ target `train`, chọn đúng `K` ảnh mỗi grade, ưu tiên bệnh nhân khác nhau.
4. Giữ cố định `selection_shots` ảnh mỗi grade, không dùng chúng để backprop.
   Phần còn lại là adaptation pool; mỗi episode chỉ tách query trong pool này.
5. RETFound encoder sinh embedding; prototype mỗi grade là trung bình embedding
   support của grade đó.
6. Fine-tune projection và block encoder cuối bằng episodic cross-entropy.
7. Chọn checkpoint và early stopping theo loss trên fixed selection, không theo
   training loss và không nhìn target test.
8. Mỗi epoch cập nhật `last.pth`; bản tốt nhất lưu `best.pth`. Resume khôi phục
   adaptation/selection split, optimizer, AMP scaler, epoch, patience và RNG.
9. Target test chỉ được đánh giá thủ công bằng `--eval-only` sau khi adaptation.

## Chạy

```powershell
python -m ai.train_fewshot.train `
  --checkpoint D:\grade\best.pth `
  --target-dataset-dir D:\dr `
  --output-dir D:\fs2 `
  --shots 10 `
  --selection-shots 2 `
  --queries 2 `
  --epochs 20
```

Resume:

```powershell
python -m ai.train_fewshot.train `
  ...cùng cấu hình và đường dẫn... `
  --resume D:\fs2\last.pth
```

Test cuối:

```powershell
python -m ai.train_fewshot.train `
  ...cùng cấu hình và đường dẫn... `
  --resume D:\fs2\best.pth `
  --eval-only
```

Folder output chỉ có ba artifact:

- `best.pth`: checkpoint có fixed-selection loss tốt nhất.
- `last.pth`: checkpoint epoch gần nhất để Resume.
- `metrics.csv`: training loss, fixed-selection loss và fixed-selection score
  từng epoch; sau `--eval-only` có thêm `before`, `after`, `delta`.
