# Hợp đồng backend tích hợp

Base URL: `/api/v1`. Tất cả endpoint ngoài đăng nhập yêu cầu bearer token theo
cơ chế hiện tại của hệ thống.

## Sàng lọc

`POST /screenings/upload` — `multipart/form-data`

Trường bắt buộc:

- `patient_id`
- `left_disc_image`
- `left_posterior_pole_image`
- `right_disc_image`
- `right_posterior_pole_image`

Trường lâm sàng tùy chọn nhưng nên thu thập:

- `visual_acuity_left`, `visual_acuity_right` (thị lực thập phân)
- `systolic_bp`, `diastolic_bp`
- `sudden_vision_loss`, `pregnant`, `kidney_disease`

Luồng lỗi:

- `400`: sai content type/tệp rỗng.
- `422`: bộ ảnh hỏng, quá nhỏ hoặc không đạt kiểm tra kỹ thuật tối thiểu.
- `503`: grading hoặc segmentation service chưa cấu hình/không phản hồi. Backend
  không sinh kết quả mock để che lỗi model.

Kết quả thành công có `left_eye` và `right_eye`, mỗi mắt gồm:

- `quality`: kết quả kỹ thuật của ảnh đĩa thị và hậu cực; luôn cần human review.
- `ai_result`: grade, confidence, probabilities, model version.
- `segmentation`: lesion list, mask URL và model version.
- `review_priority`, `follow_up_window`, `referral`, `macular_status`.
- `findings`, `actions`, `safety_flags`.

Toàn bộ phiên có `review_status: "draft"`, `guideline_ids` và disclaimer. Trường
`risk_stratification` được giữ để tương thích UI cũ nhưng giá trị là
`review_priority`, không phải thang nguy cơ y khoa.

## Báo cáo

`GET /screenings/{screening_id}/report.pdf`

Báo cáo render từ snapshot `clinical_assessment` đã lưu, không tự chạy lại rule
hoặc model. Báo cáo dự thảo ghi rõ cần bác sĩ xác nhận.

## Bác sĩ duyệt

`POST /reviews/{screening_id}`

Bác sĩ xác nhận/ghi đè grade riêng từng mắt, ghi chú và lịch tái khám. Sau khi
lưu, `screening.status = Reviewed` và `review_status = confirmed`. Mỗi mắt phải
có `image_quality` là `Good` hoặc `Fair`; ảnh `Poor` bị từ chối xác nhận và phải
chụp lại/chuyển chuyên khoa. Endpoint chỉ cho bác sĩ/chuyên khoa mắt hoặc admin.

## AI services do nhóm sở hữu

Backend dùng hai adapter cấu hình qua biến môi trường:

- `AI_GRADING_SERVICE_URL` → `POST /analyze`
- `AI_SEGMENTATION_SERVICE_URL` → `POST /segment`

Hai adapter nhận ảnh hậu cực và trường `eye`. Test dùng adapter in-memory qua
cùng interface; adapter giả không được khởi tạo trong production route.

## Các endpoint khác

- `POST /auth/login`, `GET /auth/me`
- CRUD `/patients`
- `GET /screenings/patient/{patient_id}`
- `GET /reports/epidemiology`
