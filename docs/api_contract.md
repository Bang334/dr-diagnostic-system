# Đặc Tả Hợp Đồng API (API Contract)

Tài liệu này định nghĩa giao thức và định dạng dữ liệu (JSON) trao đổi giữa các phân hệ: **Frontend (ReactJS)**, **Backend (FastAPI)** và các **Module AI Engine**.

---

## 1. Luồng Giao Tiếp Chung

```
[Frontend React] ───(1) Gửi Ảnh Võng Mạc & Mã BN ───> [Backend FastAPI]
                                                               │
                                                       (2) Gửi ảnh gốc sang AI
                                                               │
                                                               ▼
                                                       [AI Grading & Seg]
                                                               │
                                                       (3) Trả về JSON chẩn đoán
                                                               │
                                                               ▼
[Frontend React] <───(4) Trả về báo cáo tổng hợp ────── [Backend FastAPI]
```

---

## 2. API Chi Tiết Giữa Frontend Và Backend

### 2.1 Quản Lý Bệnh Nhân (Patients)
* **GET `/api/v1/patients`**
  * **Mô tả:** Lấy danh sách bệnh nhân (hỗ trợ phân trang).
  * **Response (200 OK):**
    ```json
    [
      {
        "id": 1,
        "patient_code": "BN0001",
        "full_name": "Phạm Văn Đồng",
        "gender": "Nam",
        "date_of_birth": "1965-04-12",
        "phone_number": "0912345678",
        "address": "12 Láng Hạ, Ba Đình, Hà Nội",
        "diabetes_type": "Type 2",
        "diabetes_duration_years": 8.5,
        "latest_hba1c": 7.20,
        "created_at": "2026-07-02T10:00:00+07:00",
        "updated_at": "2026-07-02T10:00:00+07:00"
      }
    ]
    ```

* **POST `/api/v1/patients`**
  * **Mô tả:** Thêm mới hồ sơ bệnh nhân.
  * **Request Body:**
    ```json
    {
      "patient_code": "BN0005",
      "full_name": "Trần Thu Hà",
      "gender": "Nữ",
      "date_of_birth": "1983-05-18",
      "phone_number": "0909090909",
      "address": "Phường Bến Nghé, Quận 1, TP. HCM",
      "diabetes_type": "Type 2",
      "diabetes_duration_years": 3.0,
      "latest_hba1c": 6.80
    }
    ```

---

### 2.2 Sàng Lọc Võng Mạc (Screenings & AI Analysis)
* **POST `/api/v1/screenings/upload`**
  * **Mô tả:** Đăng ký phiên khám và tải lên ảnh võng mạc của 2 mắt để kích hoạt phân tích AI.
  * **Request Format:** `multipart/form-data`
  * **Request Fields:**
    * `patient_id`: `1` (integer)
    * `left_eye_image`: (File ảnh mắt trái)
    * `right_eye_image`: (File ảnh mắt phải)
  * **Response (201 Created):**
    ```json
    {
      "screening_id": 12,
      "patient_code": "BN0001",
      "screening_date": "2026-07-02T18:15:30+07:00",
      "status": "AI_Analyzed",
      "left_eye": {
        "image_url": "/static/uploads/12_L.png",
        "quality": "Good",
        "ai_result": {
          "dr_grade": 2,
          "dr_label": "Moderate NPDR",
          "confidence": 0.9145,
          "probabilities": { "No_DR": 0.02, "Mild_NPDR": 0.06, "Moderate_NPDR": 0.91, "Severe_NPDR": 0.01, "Proliferative_DR": 0.00 }
        },
        "segmentation": {
          "lesion_mask_url": "/static/masks/12_L_mask.png",
          "lesions": {
            "microaneurysm": { "detected": true, "area_pct": 0.12 },
            "hemorrhage": { "detected": true, "area_pct": 0.45 },
            "hard_exudate": { "detected": false, "area_pct": 0.0 }
          }
        }
      },
      "right_eye": {
        "image_url": "/static/uploads/12_R.png",
        "quality": "Good",
        "ai_result": {
          "dr_grade": 0,
          "dr_label": "No DR",
          "confidence": 0.9820,
          "probabilities": { "No_DR": 0.98, "Mild_NPDR": 0.01, "Moderate_NPDR": 0.01, "Severe_NPDR": 0.00, "Proliferative_DR": 0.00 }
        },
        "segmentation": {
          "lesion_mask_url": null,
          "lesions": {
            "microaneurysm": { "detected": false, "area_pct": 0.0 },
            "hemorrhage": { "detected": false, "area_pct": 0.0 },
            "hard_exudate": { "detected": false, "area_pct": 0.0 }
          }
        }
      },
      "risk_stratification": "Medium",
      "clinical_recommendation": "Khám chuyên khoa mắt định kỳ trong vòng 1-2 tháng. Kiểm soát chặt chẽ đường huyết HbA1c và kiểm tra huyết áp thường xuyên."
    }
    ```

---

### 2.3 Phê Duyệt Của Bác Sĩ (Doctor Review)
* **POST `/api/v1/screenings/{screening_id}/review`**
  * **Mô tả:** Bác sĩ nhãn khoa xác nhận kết quả chẩn đoán cuối cùng, thêm ghi chú và lên lịch tái khám.
  * **Request Body:**
    ```json
    {
      "doctor_id": 2,
      "left_eye_review": {
        "final_dr_grade": 2,
        "is_agree_with_ai": true,
        "clinical_notes": "Xuất hiện vi phình mạch rải rác và một vài điểm xuất huyết nhỏ ở vùng hoàng điểm mắt trái."
      },
      "right_eye_review": {
        "final_dr_grade": 0,
        "is_agree_with_ai": true,
        "clinical_notes": "Võng mạc mắt phải bình thường."
      },
      "recall_settings": {
        "recall_in_months": 2,
        "risk_stratification": "Medium",
        "recommendation": "Tái khám đáy mắt sau 2 tháng để theo dõi tiến triển tổn thương võng mạc mắt trái."
      }
    }
    ```
  * **Response (200 OK):**
    ```json
    {
      "status": "success",
      "message": "Kết quả phê duyệt bệnh án và lịch tái khám đã được ghi nhận thành công.",
      "screening_status": "Reviewed"
    }
    ```

---

### 2.4 Báo Cáo Thống Kê Dịch Tễ (Epidemiological Reports)
* **GET `/api/v1/reports/epidemiology`**
  * **Mô tả:** Lấy dữ liệu phân phối mức độ bệnh lý DR theo độ tuổi và thời gian mắc tiểu đường.
  * **Response (200 OK):**
    ```json
    {
      "by_age_group": [
        { "age_group": "Under 40", "total_screened": 120, "dr_detected": 15, "prevalence_rate": 12.5 },
        { "age_group": "40-60", "total_screened": 648, "dr_detected": 194, "prevalence_rate": 29.93 },
        { "age_group": "Over 60", "total_screened": 480, "dr_detected": 175, "prevalence_rate": 36.46 }
      ],
      "by_diabetes_duration": [
        { "duration_group": "Under 5 years", "total_screened": 520, "dr_detected": 52, "prevalence_rate": 10.0 },
        { "duration_group": "5-10 years", "total_screened": 480, "dr_detected": 168, "prevalence_rate": 35.0 },
        { "duration_group": "Over 10 years", "total_screened": 248, "dr_detected": 164, "prevalence_rate": 66.13 }
      ]
    }
    ```
