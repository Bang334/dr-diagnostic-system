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
