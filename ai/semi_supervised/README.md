# Phân Hệ Thực Hành AI Nâng Cao (Semi-supervised & Few-shot Learning)

Thư mục này chứa mã nguồn thực hành và thử nghiệm các kỹ thuật học máy nâng cao dành cho **Thành viên 3** nhằm tối ưu hóa việc sử dụng dữ liệu ảnh võng mạc ít nhãn hoặc chưa gán nhãn trong thực tế.

---

## 📂 Các File Mã Nguồn Khung

1. **`semi_supervised_training.py`:** Chương trình PyTorch thực hành kỹ thuật **Pseudo-Labeling** (Tự gán nhãn giả). Giúp mô hình học từ cả ảnh võng mạc có nhãn (tập APTOS/EyePACS) và ảnh võng mạc chưa có nhãn thu thập tại bệnh viện.
2. **`few_shot_demo.py`:** Chương trình PyTorch thực hành kiến trúc **Prototypical Networks (ProtoNet)** phục vụ phân loại nhanh ảnh võng mạc với chỉ 1 hoặc 5 ảnh mẫu cho mỗi lớp (1-shot / 5-shot learning) có tích hợp **Episodic Training**.

---

## 🛠️ Hướng Dẫn Chuẩn Bị Dữ Liệu

### 1. Dữ liệu học bán giám sát (Semi-supervised)
Bạn cần tổ chức dữ liệu thành hai thư mục riêng biệt tại máy cục bộ hoặc server training:
```
data/dr_semi_supervised/
├── labeled/              # Thư mục chứa ảnh có nhãn (ví dụ: APTOS 2019)
│   ├── class_0/              # Không bệnh DR
│   ├── class_1/              # Mild NPDR
│   ├── class_2/              # Moderate NPDR
│   ├── class_3/              # Severe NPDR
│   └── class_4/              # Proliferative DR
└── unlabeled/            # Thư mục chứa các ảnh chụp võng mạc CHƯA gán nhãn
    ├── unlabeled_img_001.png
    ├── unlabeled_img_002.png
    └── ...
```

### 2. Thư viện yêu cầu
Để chạy mã nguồn huấn luyện, cần cài đặt thêm PyTorch và Torchvision:
```bash
pip install torch torchvision scikit-learn pandas pillow
```

---

## 🚀 Cách Chạy Thử Nghiệm

### Huấn luyện Semi-supervised (Pseudo-labeling):
```bash
python ai/semi_supervised/semi_supervised_training.py --labeled_dir data/dr_semi_supervised/labeled --unlabeled_dir data/dr_semi_supervised/unlabeled --epochs 10
```

### Chạy demo Few-shot learning (Prototypical Networks):
```bash
python ai/semi_supervised/few_shot_demo.py
```
> [!TIP]
> Bạn có thể tinh chỉnh các thông số mạng xương sống (Backbone) như đổi từ `resnet18` sang `efficientnet_b0` trong file code để đạt độ chính xác phân tầng cao hơn.
