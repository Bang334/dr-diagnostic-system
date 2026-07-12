# Luồng Xử Lý Chẩn Đoán Bệnh Võng Mạc Tiểu Đường - Toàn Hệ Thống

Tài liệu mô tả chi tiết luồng xử lý hoàn chỉnh từ giai đoạn bàn giao mô hình AI cho đến luồng chẩn đoán lâm sàng tích hợp thực tế.

---

## 1. Sơ Đồ Khái Niệm Quan Hệ Thành Viên

```
[TV1: Model Phân Loại Gốc] ───(Bàn giao file .pth)───► [ TV3: AI NÂNG CAO & HỖ TRỢ LÂM SÀNG ]
[TV2: Model Phân Đoạn Gốc] ───(Bàn giao file .pth)───►          │
                                                               │ (Tối ưu hóa Offline bằng
                                                               │  Semi-supervised & Few-shot)
                                                               ▼
                                                      [ TV3: Hệ Thống Chẩn Đoán Tích Hợp ]
                                                       - Model phân loại tối ưu
                                                       - Model phân đoạn tối ưu
                                                       - Quy tắc lâm sàng (Clinical Rules)
                                                       - Công thức phân tầng nguy cơ
```

---

## 2. Sơ Đồ Luồng Xử Lý Toàn Hệ Thống

Sơ đồ Mermaid dưới đây mô tả chi tiết luồng Online (chẩn đoán cho bệnh nhân) kết hợp với luồng Offline (tích lũy dữ liệu và tối ưu hóa mô hình do TV3 quản lý).

```mermaid
flowchart TD
    %% ═══════════════════════════════════════
    %% BƯỚC 1: ĐẦU VÀO
    %% ═══════════════════════════════════════
    DOCTOR([Bác sĩ nhãn khoa]) -->|1. Upload ảnh võng mạc| WEB[Frontend Web App]
    DOCTOR -->|1. Nhập chỉ số: HbA1c, Huyết áp, Năm ĐTĐ| WEB

    WEB -->|2. POST /analyze/full| BACKEND[Backend API - FastAPI]

    %% ═══════════════════════════════════════
    %% BƯỚC 2: QUALITY GATE
    %% ═══════════════════════════════════════
    BACKEND --> QC{3. Kiểm tra chất lượng ảnh}
    QC -->|Không đạt| ERR[Yêu cầu chụp lại]
    QC -->|Đạt chuẩn| PRE[4. Tiền Xử Lý Ảnh:<br>Resize -> Green Channel -> CLAHE -> Ben Graham]

    %% ═══════════════════════════════════════
    %% BƯỚC 3: INFERENCE VỚI CÁC MODEL ĐÃ ĐƯỢC TV3 TỐI ƯU
    %% ═══════════════════════════════════════
    PRE -->|Ảnh đã chuẩn hóa| RUN_MODELS

    subgraph TV3_RUN [Hệ Thống Chạy Mô Hình Tối Ưu Của TV3]
        direction LR
        RUN_MODELS{5. Chạy song song 2 model AI}
        RUN_MODELS -->|5a. Phân loại| M_CLASS[Model Phân Loại Đã Tối Ưu]
        RUN_MODELS -->|5b. Phân đoạn| M_SEG[Model Phân Đoạn Đã Tối Ưu]
        
        M_CLASS --> OUT_CLASS[DR Grade 0-4 + Confidence]
        M_SEG --> OUT_SEG[Lesion Mask + Diện tích %]
    end

    %% ═══════════════════════════════════════
    %% BƯỚC 4: TV3 TỔNG HỢP KẾT QUẢ & LÂM SÀNG
    %% ═══════════════════════════════════════
    OUT_CLASS --> ORCH
    OUT_SEG --> ORCH
    BACKEND -->|Chỉ số lâm sàng bệnh nhân| ORCH

    subgraph CLINICAL_DECISION [6. Bộ Xử Lý Lâm Sàng & Phân Tầng - TV3]
        direction TB
        ORCH[6a. Clinical Orchestrator<br>Gộp: AI outputs + Chỉ số lâm sàng]
        
        ORCH --> CONF_CHECK{6b. Confidence >= 70%?}
        CONF_CHECK -->|Không| WARN[Ghi nhận cảnh báo Low Confidence]
        CONF_CHECK -->|Có| RISK[6c. Tính điểm nguy cơ y tế 0-100<br>kết hợp: Grade, Tổn thương, HbA1c, Huyết áp, Năm ĐTĐ]
        WARN --> RISK

        RISK --> RULES[6d. Áp Luật Chuyển Tuyến ICO Table 3b<br>Tự động đề xuất thời hạn tái khám]
        RULES --> REPORT[6e. Sinh báo cáo lâm sàng Markdown & PDF]
    end

    %% ═══════════════════════════════════════
    %% BƯỚC 5: DUYỆT & LƯU TRỮ
    %% ═══════════════════════════════════════
    REPORT --> DRAFT[7. Bác sĩ duyệt báo cáo DRAFT]
    DRAFT --> DECIDE{Quyết định của bác sĩ}
    DECIDE -->|Đồng ý| CONFIRM[Confirm]
    DECIDE -->|Không đồng ý| OVERRIDE[Override: Sửa Grade / Ghi chú]

    CONFIRM --> DB[(Database PostgreSQL)]
    OVERRIDE --> DB
    
    DB --> RECALL[8. Lên lịch hẹn tái khám tự động]
    DB --> PDF_DL[8. Tải PDF báo cáo chẩn đoán]

    %% ═══════════════════════════════════════
    %% BƯỚC 6: CHU TRÌNH TỐI ƯU HÓA OFFLINE (TV3 thực hiện)
    %% ═══════════════════════════════════════
    DB -.->|Tích lũy ảnh chưa nhãn| OFFLINE_OPTIMIZE

    subgraph OFFLINE_OPTIMIZE [9. Chu Trình Tối Ưu Hóa Offline - TV3]
        direction TB
        
        INPUT_MODELS[Nhận Model Gốc TV1/TV2]
        
        SEMI[9a. Semi-supervised Learning:<br>Dự đoán nhãn giả cho ảnh chưa nhãn -> Lọc confidence >= 95% -> Retrain]
        FEW[9b. Few-shot Learning:<br>Tích hợp 1-5 ảnh ca hiếm -> Tính Prototype đặc trưng]
        
        INPUT_MODELS --> SEMI
        INPUT_MODELS --> FEW
    end

    SEMI -.->|Cập nhật trọng số weights| M_CLASS
    SEMI -.->|Cập nhật trọng số weights| M_SEG
    FEW -.->|Bổ sung Prototype nhận diện| M_CLASS
```

---

## 3. Mô Tả Chi Tiết Luồng Chẩn Đoán Tích Hợp

### Bước 1: Thu Thập Thông Tin
Bác sĩ nhãn khoa tải lên ảnh võng mạc của bệnh nhân và nhập các thông tin lâm sàng bao gồm:
- **HbA1c (%):** Đánh giá khả năng kiểm soát đường huyết của bệnh nhân đái tháo đường.
- **Huyết áp (mmHg):** Huyết áp cao làm tăng nguy cơ tổn thương mạch máu võng mạc.
- **Số năm bị đái tháo đường:** Thời gian mắc bệnh càng lâu, nguy cơ xuất hiện biến chứng võng mạc càng cao.

### Bước 2: Quality Gate & Tiền Xử Lý Ảnh
- Thu đủ hai ảnh mỗi mắt: đĩa thị và hậu cực/hoàng điểm.
- Hệ thống loại ảnh hỏng/quá nhỏ, xóa metadata khi tái mã hóa và gắn trạng thái
  `ReviewRequired`; người đọc xác nhận gradability. Tiền xử lý cụ thể thuộc model
  và phải giống quy trình đã validation, không mặc định CLAHE làm ảnh đạt chuẩn.

### Bước 3: Inference qua adapter model có quản lý phiên bản
Hệ thống gọi model qua adapter HTTP; nếu model không sẵn sàng thì trả `503`, không sinh mock:
1. **Model phân loại (TV1 gốc đã tối ưu):** Dự đoán cấp độ DR Grade từ 0 đến 4 cùng độ tin cậy (Confidence).
2. **Model phân đoạn (TV2 gốc đã tối ưu):** Định vị các vùng tổn thương (Microaneurysm, Hemorrhage, Hard Exudate) và tính tỷ lệ diện tích tổn thương trên ảnh.

### Bước 4: Hỗ trợ sàng lọc và ưu tiên bác sĩ rà soát
Clinical Analysis Module tổng hợp dữ liệu thành dự thảo:
- **Kiểm tra độ tin cậy:** Nếu `confidence` < 70%, hệ thống gắn cờ cảnh báo để bác sĩ cẩn trọng khi duyệt.
- **Không tính risk score y khoa tự đặt.** Chỉ có `review_priority` để sắp hàng đợi.
- **Mốc tham khảo:** Grade 3 chuyển chuyên khoa và đánh giá không quá 3 tháng;
  Grade 4 chuyển tuyến tỉnh/trung ương dưới 1 tháng. Grade 0–2 do bác sĩ cá thể hóa.
- **Hoàng điểm:** hard exudate chỉ kích hoạt đánh giá hoàng điểm; không kết luận
  DME hoặc chỉ định anti-VEGF/laser nếu thiếu vị trí dày võng mạc/OCT.
- Giảm thị lực đột ngột được chuyển khám trong ngày, không chờ AI; thị lực dưới
  5/10 được chuyển chuyên khoa kể cả ảnh không thấy DR.
- **Sinh báo cáo:** Tạo báo cáo tóm tắt Markdown và xuất PDF nhúng kèm ảnh võng mạc minh chứng.

### Bước 5: Duyệt Kết Quả & Database
- Bác sĩ xem báo cáo nháp (Draft) trên Web.
- Bác sĩ có quyền phê duyệt kết quả AI (Confirm) hoặc thay đổi mức độ chẩn đoán theo kinh nghiệm lâm sàng cá nhân (Override).
- Dữ liệu được lưu trữ chính thức vào PostgreSQL để tự động đặt lịch nhắc hẹn tái khám cho bệnh nhân.

### Bước 6: Nghiên cứu offline, tách khỏi inference
Pseudo-labeling/Few-shot chỉ chạy trong môi trường nghiên cứu với consent/quản trị
dữ liệu phù hợp. Model mới phải vượt baseline trên test set giữ kín, được hiệu
chỉnh, bác sĩ phê duyệt và quản lý phiên bản trước khi adapter production sử dụng.
Không tự động học từ database hoặc deploy chỉ dựa trên confidence.

---

## 4. Bảng Phân Chia Vai Trò Trong Dự Án

| | Thành Viên 1 | Thành Viên 2 | Thành Viên 3 (Tích hợp & AI nâng cao) |
|---|---|---|---|
| **Công việc chính** | Xây dựng mô hình phân loại DR Grade | Xây dựng mô hình phân đoạn tổn thương | Tối ưu hóa mô hình + Phát triển hệ thống luật chẩn đoán y khoa |
| **Bàn giao** | File model gốc (`.pth`) + API phân loại cơ bản | File model gốc (`.pth`) + API phân đoạn cơ bản | Vận hành toàn bộ pipeline chẩn đoán (Inference, Rules, PDF, DB, Web) |
| **Thuật toán AI áp dụng** | Supervised Learning (EfficientNet, ResNet, ConvNeXt) | Supervised Learning (U-Net, Attention U-Net) | Semi-supervised Learning (Pseudo-labeling) + Few-shot Learning (ProtoNet) |
