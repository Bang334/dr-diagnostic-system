# Semi-supervised và few-shot với RETFound

Hai pipeline trong thư mục này là thí nghiệm nghiên cứu dùng lại
`checkpoint-best.pth` của mô hình grading RETFound. Chúng không tự tải backbone
mới và không được đưa thẳng vào inference lâm sàng.

Notebook Colab: `Semi_Supervised_Few_Shot_Colab.ipynb`.

## Dữ liệu được phép sử dụng

Dataset có nhãn phải giữ nguyên cấu trúc split:

```text
fundus_dataset/
├── train/       # hoặc training/
│   ├── 0/ ... 4/
├── validation/  # hoặc val/
│   ├── 0/ ... 4/
└── test/
    ├── 0/ ... 4/
```

- Semi-supervised chỉ tối ưu trên `train` và ảnh ngoài chưa có nhãn.
- Few-shot chọn đúng `K` ảnh mỗi lớp từ `train` làm support cố định; `val` không
  được dùng để tránh tăng ngân sách nhãn target.
- Semi-supervised không nạp `test`. Few-shot chỉ đánh giá `test` target trước và
  sau thích nghi; không dùng metric test để train, early stopping hoặc chọn
  checkpoint.
- Thư mục ảnh chưa nhãn phải là nguồn ngoài, không được trỏ vào `train`, `val`
  hoặc `test`. Script chặn các đường dẫn bị trùng.
- Nếu dữ liệu có hai mắt của cùng bệnh nhân, split phải được tạo theo bệnh nhân
  trước khi chạy các thí nghiệm này.

## Cài dependencies

```powershell
pip install -r ai/grading/requirements-train.txt
pip install -r ai/semi_supervised/requirements-research.txt
```

## Pseudo-labeling

Pipeline nạp checkpoint CE tốt nhất, đánh pseudo-label một lần bằng transform
validation, giữ ảnh có confidence từ `0.95`, giới hạn số ảnh mỗi lớp, rồi
fine-tune bằng:

- ảnh có nhãn với trọng số `1.0`;
- ảnh pseudo-label với trọng số mặc định `0.25 × confidence`;
- LR head `1e-5`, LR backbone `1e-6`;
- chọn checkpoint theo QWK validation.

Đây là self-training có **labeled replay**: model student khởi tạo từ best
checkpoint rồi học trên hợp của `train` cũ và pseudo-label mới. Dataset grading
cũ không phải nguồn unlabeled; nó được phát lại để giảm confirmation bias và
catastrophic forgetting.

```powershell
python -m ai.semi_supervised.semi_supervised_training `
  --checkpoint D:\checkpoints\checkpoint-best.pth `
  --dataset-dir D:\data\fundus_merged `
  --unlabeled-dir D:\data\fundus_unlabeled `
  --output-dir D:\runs\retfound_pseudo_v1
```

Các artifact chính:

- `pseudo_labels.csv`: đường dẫn, nhãn giả và confidence để audit;
- `checkpoint-best.pth`: luôn ít nhất bằng trọng số parent nếu fine-tune không
  cải thiện QWK;
- `checkpoint-last.pth`: có optimizer/scheduler của run nghiên cứu;
- `history.jsonl` và `summary.json`.

Nếu không ảnh nào vượt threshold, script dừng thay vì tự hạ ngưỡng. Hãy kiểm tra
calibration và domain ảnh ngoài trước khi chọn ngưỡng thấp hơn.

## Few-shot target-domain adaptation

`few_shot_demo.py` mô phỏng tình huống model nguồn được huấn luyện tại một quốc
gia/bệnh viện rồi thích nghi tại nơi mới bằng rất ít nhãn. Dataset target phải
hoàn toàn chưa xuất hiện khi train checkpoint nguồn và vẫn có năm DR grade
`0..4`.

Script chọn support một lần, đúng `K` ảnh/lớp, lưu vào `support_manifest.csv` và
chỉ tạo leave-one-out episode trong tập cố định này. Mặc định embedding RETFound
được giữ nguyên chiều và chỉ block transformer cuối cùng cùng các lớp norm được
mở. Target `val` không tham gia chọn model; target `test` chỉ báo cáo:

1. classifier nguồn chưa dùng nhãn target;
2. ProtoNet dùng support nhưng chưa gradient adaptation;
3. ProtoNet sau gradient adaptation trên support.

```powershell
python -m ai.semi_supervised.few_shot_demo `
  --checkpoint D:\checkpoints\checkpoint-best.pth `
  --target-dataset-dir D:\data\foreign_hospital_target `
  --output-dir D:\runs\retfound_target_5shot_seed42 `
  --shots 5 `
  --queries 1 `
  --seed 42
```

Các artifact chính:

- `support_manifest.csv`: bằng chứng model chỉ thấy đúng `K × 5` ảnh target;
- `comparison.json`: metric trước/sau trên cùng target test;
- `checkpoint-adapted-protonet.pth`: lưu encoder, projection và prototype;
- `history.jsonl` và `summary.json`: loss support và kiểm tra chống test leakage.

Checkpoint đã lưu prototype nên không cần đọc lại support tại inference, nhưng
chưa tương thích trực tiếp với API grading năm lớp. Không sao chép artifact này
vào `ai/weights` production trước khi có adapter inference và kiểm định nhiều
seed/domain.

## Chạy trên Colab/Drive

Notebook sẽ:

1. mount Google Drive;
2. clone/pull đúng branch;
3. dò các file `checkpoint-best.pth` trên Drive;
4. với semi, mặc định tự tải dataset fundus gộp từ Kaggle làm labeled replay;
   với few-shot, bắt buộc nhập dataset target mới có split cố định;
5. nhận nguồn ảnh chưa nhãn từ `kaggle:owner/dataset`, thư mục hoặc ZIP khi chạy
   semi-supervised;
6. lưu mỗi phương pháp vào output directory riêng trên Drive và hiển thị
   `support_manifest.csv`/`comparison.json` sau run few-shot.

Script từ chối output directory đã có artifact để tránh nối lẫn hai run hoặc
ghi đè checkpoint cũ. Khi chạy lại, hãy đổi `Tên run` trong notebook.

Checkpoint đầu vào phải là checkpoint CE do `ai/grading/train.py` tạo. Notebook
không cần đăng nhập Hugging Face vì kiến trúc được dựng với `pretrained=False`
và toàn bộ trọng số được lấy từ checkpoint Drive.

Để tự tải dữ liệu, thêm `KAGGLE_API_TOKEN` vào Colab Secrets. Dataset có nhãn
mặc định là `sehastrajits/fundus-aptosddridirdeyepacsmessidor`. Semi-supervised
vẫn cần một Kaggle dataset khác hoặc một thư mục ảnh ngoài làm nguồn chưa nhãn;
không được dùng lại dataset có nhãn hay test split.
