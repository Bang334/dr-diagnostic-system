# Semi-supervised / Few-shot — trạng thái nghiên cứu

Mã scaffold đã được chuyển vào chính thư mục này nhưng chưa được đưa vào
inference của hệ thống chính. Lý do: workspace không có dataset,
checkpoint, patient-level split hoặc báo cáo metric trên test set giữ kín.

- `semi_supervised_training.py`: pseudo-labeling scaffold; chưa có supervised
  baseline, calibration hoặc kết quả trên ảnh thật.
- `few_shot_demo.py`: ProtoNet chạy trên tensor giả lập; không phải thí nghiệm
  few-shot y khoa.

Tiêu chí nghiệm thu và tích hợp model được ghi tại
`docs/tv3_integration_status.md`. Chỉ model đã validation và có version mới được
triển khai sau `AI_GRADING_SERVICE_URL`/`AI_SEGMENTATION_SERVICE_URL`.

Không sao chép checkpoint thử nghiệm vào backend và không tự học lại từ dữ
liệu bệnh nhân production. Hai script được giữ để tái lập nghiên cứu, không phải
model production.

## Môi trường nghiên cứu

```powershell
pip install -r ai/semi_supervised/requirements-research.txt
```

## Cấu trúc dữ liệu pseudo-labeling

```text
data/dr_semi_supervised/
├── labeled/class_0 ... class_4/
└── unlabeled/
```

Chạy scaffold:

```powershell
python ai/semi_supervised/semi_supervised_training.py --labeled_dir data/dr_semi_supervised/labeled --unlabeled_dir data/dr_semi_supervised/unlabeled
python ai/semi_supervised/few_shot_demo.py
```

Lệnh thứ hai chỉ chạy tensor giả lập. Không công bố accuracy của demo như kết
quả trên ảnh võng mạc.
