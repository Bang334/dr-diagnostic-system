# Hướng dẫn lâm sàng và giới hạn của module AI

> Module chỉ hỗ trợ **sàng lọc và ưu tiên bác sĩ đọc ảnh**. Kết quả luôn là
> `draft`; không phải chẩn đoán, đơn thuốc hay chỉ định điều trị tự động.

## Nguồn áp dụng

1. Bộ Y tế, Quyết định **2558/QĐ-BYT ngày 20/9/2022**, Hướng dẫn chẩn đoán,
   điều trị và quản lý bệnh võng mạc đái tháo đường:
   https://kcb.vn/upload/2005611/20230622/Quyet_dinh_huong_dan_benh_VMDTD15-9-2022_signed_d80d3.pdf
2. Bộ Y tế, Quyết định **2557/QĐ-BYT ngày 20/9/2022**, Quy trình chụp ảnh đáy
   mắt không huỳnh quang:
   https://kcb.vn/upload/2005611/20230622/Quyet_dinh_Quy_trinh_chup_anh_day_mat15-9-2022_signed_354d2.pdf
3. ICDR, Wilkinson et al. 2003, thang phân loại DR năm mức.
4. ADA Standards of Care in Diabetes 2026, mục Retinopathy.

## Rule được phép dùng

| Đầu ra | Ý nghĩa |
| --- | --- |
| Grade 0–4 | Dự đoán ICDR của model, cần người đọc xác nhận riêng từng mắt. |
| `review_priority` | Thứ tự hàng đợi nội bộ, không phải risk score y khoa. |
| Grade 3 | Chuyển chuyên khoa, đánh giá không quá 3 tháng và sớm hơn theo bệnh cảnh. |
| Grade 4 | Chuyển chuyên khoa tuyến tỉnh/trung ương dưới 1 tháng. |
| Thị lực < 5/10 | Chuyển chuyên khoa kể cả ảnh không thấy DR. |
| Giảm thị lực đột ngột | Khám trực tiếp trong ngày, không chờ AI. |

Khoảng Grade 0–2 trong hệ thống là lời nhắc bảo thủ để bác sĩ cá thể hóa, không
phải lịch hẹn cứng. Thai kỳ, bệnh thận, kiểm soát đường huyết/huyết áp, triệu
chứng, thị lực và khả năng theo dõi đều có thể làm thay đổi lịch.

## Điều hệ thống không được suy luận

- Không chẩn đoán DME từ diện tích hard exudate. Center involvement cần dày
  võng mạc, vị trí so với hố trung tâm và/hoặc OCT.
- Không gọi “high-risk PDR” từ tổng diện tích hemorrhage. Cần nhận diện tân
  mạch, xuất huyết trước võng mạc/dịch kính và bác sĩ xác nhận.
- Không dùng confidence 70%, diện tích lesion hoặc điểm cộng HbA1c/huyết áp như
  ngưỡng y khoa. Confidence 70% chỉ là cờ vận hành và phải hiệu chỉnh theo bộ
  validation tại cơ sở triển khai.
- Không tự chỉ định anti-VEGF, PRP, laser hoặc phẫu thuật.

## Yêu cầu pháp lý/vận hành Việt Nam

- Luật Khám bệnh, chữa bệnh 15/2023/QH15: quyết định chuyên môn thuộc người
  hành nghề/cơ sở đủ điều kiện.
- Nghị định 98/2021/NĐ-CP, sửa đổi bởi Nghị định 07/2023/NĐ-CP: phải thực hiện
  đánh giá phân loại, hồ sơ và thủ tục thiết bị y tế phù hợp với mục đích sử dụng;
  tài liệu này không tự kết luận hệ thống thuộc loại B hay C.
- Nghị định 13/2023/NĐ-CP: ảnh mắt và hồ sơ sức khỏe là dữ liệu nhạy cảm; triển
  khai thật cần căn cứ xử lý, phân quyền, nhật ký, mã hóa, thời hạn lưu/xóa và hồ
  sơ đánh giá tác động xử lý dữ liệu cá nhân.
- Trước sử dụng thật cần validation đa trung tâm, calibration theo máy/cơ sở,
  giám sát drift, quản lý phiên bản model, quy trình sự cố và bác sĩ nhãn khoa
  phê duyệt toàn bộ rule.
