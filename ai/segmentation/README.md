# Phân Hệ Phân Đoạn Tổn Thương Võng Mạc (Retinal Lesion Segmentation Engine)

Thư mục này chứa mã nguồn nghiên cứu, định nghĩa kiến trúc và huấn luyện các mô hình **Attention U-Net** phục vụ phân đoạn 3 loại tổn thương võng mạc tiểu đường (DR):
* **Microaneurysm (MA - Vi phình mạch)**: Nhãn chuẩn IDRiD, màu overlay Đỏ.
* **Hemorrhage (HE - Xuất huyết)**: Nhãn chuẩn IDRiD, màu overlay Cam.
* **Hard Exudate (EX - Tiết cứng)**: Nhãn chuẩn IDRiD, màu overlay Vàng.

---

## 📂 Cấu Trúc Thư Mục

* `/configs/`: Cấu hình siêu tham số huấn luyện `RetinalConfig`.
* `/models/`: Định nghĩa kiến trúc `PureAttentionUNet` và `smp.Unet` (ResNet34 / ResNet50 backbone).
* `/trainer/`: PyTorch Lightning Trainer Module (`RetinalLesionLightningModule`).
* `/losses/`: Hàm mất mát kết hợp `ComboLoss` (BinaryFocalLoss + TverskyLoss).
* `/metrics/`: Đo lường hiệu năng Dice, IoU, AUPRC, HD95, ASSD (`eval_metrics.py`).
* `/preprocessing/`: Tiền xử lý ảnh ROI crop, Green channel extraction, CLAHE, Ben Graham.
* `/augmentation/`: Tăng cường dữ liệu Copy-Paste Augmentation và Albumentations.

---

## 🔗 Tích hợp với Backend

Thư mục này được gọi trực tiếp bởi `backend/app/services/lesion_inference.py` thông qua cấu hình biến môi trường `SEGMENTATION_LESION_PATH=ai/segmentation` trong tệp `backend/.env`.

---

## ⬇️ Hướng Dẫn Tải Checkpoints Mô Hình (Model Weights)

Vì giới hạn kích thước tệp của GitHub, các tệp trọng số Attention U-Net (`.ckpt`) KHÔNG được push trực tiếp lên repository.

### 1. Liên kết tải về:
* 📥 **Google Drive Folder:** [Tải bộ Checkpoints Attention U-Net tại đây](https://drive.google.com/drive/folders/1RvJSFmdrHxIxBLVQTIiECiQA5Ye817yf?usp=sharing)

### 2. Cấu trúc đặt tệp trên máy cá nhân:
Tải 3 tệp `.ckpt` từ đường link trên và đặt vào thư mục `backend/checkpoints/` theo đúng đường dẫn:

```text
backend/checkpoints/
├── idrid_MA/
│   └── best-checkpoint-epoch=17-val_dice=0.0305.ckpt
├── idrid_HE/
│   └── best-checkpoint-epoch=70-val_dice=0.0272.ckpt
└── idrid_EX/
    └── best-checkpoint-epoch=64-val_dice=0.0414.ckpt
```

Khi khởi chạy Backend, hệ thống sẽ tự động nạp trước (Pre-warm) 3 mô hình này vào RAM/VRAM để thực hiện suy luận phân đoạn tổn thương (MA, HE, EX) siêu tốc.

