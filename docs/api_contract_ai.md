# Đặc Tả API Contract AI Engine (Thành Viên 1 & 2)

Tài liệu này định nghĩa giao thức và định dạng dữ liệu (JSON Response) giữa **Backend FastAPI** và các **Module AI Engine** (Phân loại của TV1 & Phân đoạn của TV2). Thành viên 3 tích hợp các API này trong Clinical Orchestrator để đưa ra khuyến nghị lâm sàng.

---

## 1. API Phân Loại Cấp Độ DR (Thành Viên 1)

*   **Endpoint:** `POST /analyze`
*   **Content-Type:** `multipart/form-data`
*   **Tham số đầu vào:**
    *   `file`: File ảnh võng mạc đáy mắt (`PNG`, `JPEG`, hoặc `JPG`).

### Response JSON bắt buộc:
```json
{
  "dr_grade": 3,
  "dr_label": "Severe NPDR",
  "confidence": 0.9105,
  "probabilities": {
    "No DR": 0.0152,
    "Mild NPDR": 0.0241,
    "Moderate NPDR": 0.0502,
    "Severe NPDR": 0.9105,
    "Proliferative DR": 0.0000
  },
  "model_version": "efficientnet_b3_v1.0",
  "preprocessed_preview_b64": "data:image/png;base64,iVBOR..." // (Tùy chọn) Ảnh base64 sau khi tiền xử lý Ben Graham
}
```

---

## 2. API Phân Đoạn Tổn Thương Võng Mạc (Thành Viên 2)

*   **Endpoint:** `POST /segment`
*   **Content-Type:** `multipart/form-data`
*   **Tham số đầu vào:**
    *   `file`: File ảnh võng mạc đáy mắt (`PNG`, `JPEG`, hoặc `JPG`).

### Response JSON bắt buộc:
```json
{
  "lesions": [
    {
      "key": "microaneurysm",
      "detected": true,
      "area_pct": 0.25,
      "confidence": 0.78
    },
    {
      "key": "hemorrhage",
      "detected": true,
      "area_pct": 0.65,
      "confidence": 0.82
    },
    {
      "key": "hard_exudate",
      "detected": true,
      "area_pct": 0.35,
      "confidence": 0.71
    },
    {
      "key": "soft_exudate",
      "detected": false,
      "area_pct": 0.0,
      "confidence": null
    }
  ],
  "overlay_b64": "data:image/png;base64,iVBOR...",
  "model_version": "attention_unet_v1.2",
  "status": "ok"
}
```

*Lưu ý:* Danh sách `lesions` bắt buộc phải trả về đúng 4 keys: `microaneurysm`, `hemorrhage`, `hard_exudate`, `soft_exudate` để Clinical Orchestrator phân tích và gộp đúng tổn thương cho thuật toán y khoa.
