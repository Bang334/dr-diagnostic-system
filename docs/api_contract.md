# Hợp đồng backend tích hợp

Base URL: `/api/v1`. Tất cả endpoint ngoài đăng nhập yêu cầu bearer token theo
cơ chế hiện tại của hệ thống.

## Sàng lọc

`POST /screenings/upload` — `multipart/form-data`

Trường bắt buộc:

- `patient_id`
- Ít nhất một trong hai trường `left_fundus_image`, `right_fundus_image`.

Loại đái tháo đường, thời gian mắc bệnh, HbA1c, tuổi và giới tính được lấy từ
hồ sơ bệnh nhân đã chọn; frontend không gửi lại các trường này trong form ảnh.

Luồng lỗi:

- `400`: sai content type/tệp rỗng.
- `413`: ảnh vượt quá giới hạn dung lượng cấu hình.
- `422`: bộ ảnh hỏng, quá nhỏ hoặc không đạt kiểm tra kỹ thuật tối thiểu.
- `503`: grading hoặc segmentation service chưa cấu hình/không phản hồi. Backend
  không sinh kết quả mock để che lỗi model.

Kết quả thành công có `left_eye` và/hoặc `right_eye` tương ứng ảnh đã gửi. Mỗi mắt gồm:

- `quality`: kết quả kỹ thuật của ảnh fundus; luôn cần human review.
- `ai_result`: grade, confidence, probabilities, model version.
- `segmentation`: lesion list, mask URL và model version.
- `review_priority`, `follow_up_window`, `referral`.
- `findings`, `actions`, `safety_flags`.

Toàn bộ phiên có `clinical_summary` do Gemini hoặc quy tắc dự phòng soạn từ dữ
liệu đã loại bỏ định danh trực tiếp, `rule_summary` do quy tắc backend tạo độc lập,
`guideline_ids` và disclaimer. Cả hai bản đều ghi rõ mốc tái khám theo từng mắt;
Gemini không được tự thay đổi `follow_up_window` do backend chỉ định. Phần tổng hợp
chỉ trả về trong response hiện tại, không được lưu lặp lại vào bảng `screenings`. Trường
`risk_stratification` được giữ để tương thích UI cũ nhưng giá trị là
`review_priority`, không phải thang nguy cơ y khoa.

Mỗi summary có thêm ba trường hỗ trợ bác sĩ rà soát:

- `diagnostic_impression`: nhận định hỗ trợ chẩn đoán **DR**, không phải chẩn
  đoán đái tháo đường.
- `diagnostic_basis`: grade, confidence và bằng chứng tổn thương thực sự có trong
  output mô hình.
- `diagnostic_limitations`: giới hạn dữ liệu/model và yêu cầu bác sĩ/xét nghiệm
  xác nhận phù hợp.

## Bác sĩ duyệt

`POST /reviews/{screening_id}`

Bác sĩ xác nhận/ghi đè grade riêng từng mắt, ghi chú và lịch tái khám. Sau khi
lưu, `screening.status = Reviewed`. Endpoint chỉ cho bác sĩ/chuyên khoa mắt hoặc admin.

## Lưu trữ kết quả

- `screenings`: liên kết bệnh nhân/bác sĩ, tối đa hai URL ảnh fundus và trạng thái.
- `ai_results`: grade, confidence, probabilities và phiên bản mô hình theo từng mắt.
- `lesion_segmentation_results`: nhãn tổn thương, diện tích và mask theo từng mắt.

Database không lưu ảnh đĩa thị riêng, điểm chất lượng ảnh, `review_status` hoặc
snapshot JSON `clinical_assessment` trong bảng `screenings`.

## AI services do nhóm sở hữu

Backend dùng hai adapter cấu hình qua biến môi trường:

- `AI_GRADING_SERVICE_URL` → `POST /analyze`
- `AI_SEGMENTATION_SERVICE_URL` → `POST /segment`

Hai adapter nhận ảnh fundus và trường `eye`. Test dùng adapter in-memory qua
cùng interface; adapter giả không được khởi tạo trong production route.

## Các endpoint khác

- `POST /auth/login`, `GET /auth/me`
- CRUD `/patients`
- `GET /screenings/patient/{patient_id}`
- `GET /reports/epidemiology`
