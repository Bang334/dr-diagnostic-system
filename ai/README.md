# Phân Hệ Học Sâu - Trí Tuệ Nhân Tạo (AI Engine)

Thư mục này chứa mã nguồn phục vụ việc nghiên cứu, tiền xử lý và huấn luyện các mô hình học sâu ứng dụng trong phân tích ảnh võng mạc đáy mắt (Fundus Image).

---

## 📂 Phân Chia Thư Mục Nghiên Cứu

* `/preprocessing/`: Các kỹ thuật xử lý ảnh võng mạc chuyên biệt trước khi đưa vào mô hình học sâu:
  * **Green Channel Extraction:** Trích xuất kênh màu xanh lá (nơi có độ tương phản mạch máu và tổn thương rõ nhất).
  * **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Cân bằng độ tương phản cục bộ để làm rõ các vi phình mạch và vùng xuất huyết mờ.
  * **Ben Graham Preprocessing:** Kỹ thuật loại bỏ nhiễu ánh sáng nền bằng cách trừ ảnh mờ Gaussian và chuẩn hóa kích thước hình tròn của võng mạc.
* `/grading/`: Huấn luyện mô hình phân loại 5 mức độ bệnh lý võng mạc tiểu đường (No DR, Mild, Moderate, Severe, Proliferative DR) theo thang ICDR sử dụng các kiến trúc:
  * **EfficientNet (B0 - B4)**
  * **ResNet (50, 101)**
  * **ConvNeXt**
* `/segmentation/`: Huấn luyện mô hình phân đoạn tổn thương (Lesion Segmentation) phát hiện vị trí các vùng tổn thương (Microaneurysm, Hemorrhage, Hard Exudate) sử dụng kiến trúc:
  * **U-Net** và các biến thể (Attention U-Net, ResUNet).
* `/semi_supervised/`: Nghiên cứu kỹ thuật học bán giám sát (Semi-supervised Learning) và Few-shot Learning giúp tận dụng dữ liệu lâm sàng chưa gán nhãn lớn tại Việt Nam.
* `/weights/`: Thư mục lưu trữ trọng số mô hình đã huấn luyện (Được bỏ qua bởi `.gitignore`, không push lên GitHub).

---

## 🛠️ Quy Chuẩn Dữ Liệu Đầu Ra (API Contract)

Để phân hệ AI có thể tích hợp mượt mà với Backend (FastAPI) và Frontend, các API phục vụ inference của mô hình AI cần tuân thủ định dạng JSON sau:

### 1. API Phân Loại Mức Độ (DR Grading)
* **Endpoint:** `POST /api/v1/ai/grading`
* **Input:** Ảnh võng mạc gốc (multipart/form-data)
* **Output JSON:**
```json
{
  "status": "success",
  "model_version": "efficientnet_b3_v1.0",
  "dr_grade": 2,
  "dr_label": "Moderate NPDR",
  "confidence": 0.9245,
  "probabilities": {
    "No_DR": 0.0120,
    "Mild_NPDR": 0.0535,
    "Moderate_NPDR": 0.9245,
    "Severe_NPDR": 0.0080,
    "Proliferative_DR": 0.0020
  }
}
```

### 2. API Phân Đoạn Tổn Thương (Lesion Segmentation)
* **Endpoint:** `POST /api/v1/ai/segment`
* **Input:** Ảnh võng mạc gốc (multipart/form-data)
* **Output JSON:**
```json
{
  "status": "success",
  "model_version": "unet_lesions_v1.0",
  "lesion_mask_url": "/static/masks/segmentation_result_102.png", // Ảnh mask transparent vẽ đè lên ảnh gốc
  "lesions": {
    "microaneurysm": { "detected": true, "area_pct": 0.1250 },
    "hemorrhage": { "detected": true, "area_pct": 0.8520 },
    "hard_exudate": { "detected": false, "area_pct": 0.0000 }
  }
}
```

---

## 📈 Bộ Dữ Liệu Huấn Luyện Khuyến Nghị (Public Datasets)

1. **EyePACS (Diabetic Retinopathy Detection):** Kho dữ liệu khổng lồ trên Kaggle chứa 35,000+ ảnh võng mạc phục vụ huấn luyện mô hình phân loại DR Grade.
2. **APTOS 2019 (Kaggle):** Bộ dữ liệu chất lượng cao gồm 3,662 ảnh võng mạc được gán nhãn chuẩn bởi các bác sĩ lâm sàng, phù hợp để tinh chỉnh (fine-tune) mô hình Grading.
3. **IDRiD (Indian Diabetic Retinopathy Image Dataset):** Bộ dữ liệu chuẩn duy nhất cung cấp nhãn phân đoạn (segmentation pixel-level ground truth) cho từng loại tổn thương (Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates). Rất phù hợp để huấn luyện mô hình U-Net.
