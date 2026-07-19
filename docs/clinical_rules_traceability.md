# Truy vết rule lâm sàng đã tích hợp

Mọi đầu ra của `backend/app/clinical` là dự thảo hỗ trợ sàng lọc và phải được
bác sĩ/chuyên khoa mắt xác nhận.

| Rule/Interface | Nguồn hoặc loại | Cách dùng an toàn |
| --- | --- | --- |
| ICDR Grade 0–4 | ICDR; QĐ 2558/QĐ-BYT | Dự đoán riêng từng mắt; Grade 3/4 luôn cần specialist confirmation. |
| Nhận định hỗ trợ chẩn đoán DR | QĐ 2558/QĐ-BYT; ICDR | Kết hợp grade, confidence và tổn thương quan sát được; luôn nêu giới hạn và cần bác sĩ xác nhận. |
| Không thấy DR trên ảnh | QĐ 2558/QĐ-BYT; ADA 2026 | Không được diễn giải thành không mắc đái tháo đường; nếu nghi ngờ đái tháo đường phải dùng xét nghiệm chuẩn. |
| Hai ảnh mỗi mắt | QĐ 2557/QĐ-BYT | Ảnh đĩa thị và hậu cực/hoàng điểm; đúng người, đúng mắt. |
| Thị lực < 5/10 | QĐ 2558/QĐ-BYT | Chuyển chuyên khoa kể cả ảnh không phát hiện DR. |
| Grade 3 | QĐ 2558/QĐ-BYT | Chuyển chuyên khoa, đánh giá không quá 3 tháng và sớm hơn theo bệnh cảnh. |
| Grade 4 | QĐ 2558/QĐ-BYT | Chuyển tuyến tỉnh/trung ương dưới 1 tháng. |
| Hard exudate | Internal safety trigger | Chỉ yêu cầu đánh giá hoàng điểm/OCT; không kết luận DME. |
| Confidence < 0,70 | Operational heuristic | Tăng ưu tiên bác sĩ đọc/chụp lại; không phải ngưỡng y khoa. |
| `review_priority` | Internal workflow | Sắp hàng đợi, không trình bày như risk score lâm sàng. |
| HbA1c > 8%, BP ≥ 140/90 | Operational flags | Nhắc bác sĩ đánh giá và cá thể hóa; không cộng điểm hoặc tự đổi điều trị. |
| Giảm thị lực đột ngột | Safety override | Khám trực tiếp trong ngày, không chờ AI. |

## Rule đã loại bỏ

- `hard_exudate.area_pct >= 0.1` suy ra DME/center-involved DME.
- `hemorrhage.area_pct >= 0.5` suy ra high-risk PDR hoặc mốc 24–48 giờ.
- Risk score 0–100 từ trọng số lesion, HbA1c, huyết áp và thời gian mắc bệnh.
- Tự gắn chất lượng ảnh `Good` chỉ từ định dạng/kích thước file.
- Tự chỉ định anti-VEGF, PRP, laser hoặc phẫu thuật.

## Nguồn chính

- QĐ 2557/QĐ-BYT ngày 20/9/2022.
- QĐ 2558/QĐ-BYT ngày 20/9/2022.
- Wilkinson et al., ICDR disease severity scale, 2003.
- ADA Standards of Care in Diabetes 2026, Retinopathy.
- Luật Khám bệnh, chữa bệnh 15/2023/QH15; Luật Bảo vệ dữ liệu cá nhân
  91/2025/QH15.
- Nghị định 13/2023/NĐ-CP; Nghị định 98/2021/NĐ-CP sửa đổi bởi 07/2023/NĐ-CP.
