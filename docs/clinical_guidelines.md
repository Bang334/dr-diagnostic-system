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

## Ranh giới giữa chẩn đoán đái tháo đường và chẩn đoán DR

- QĐ 2558/QĐ-BYT xác định bệnh võng mạc đái tháo đường ở người đã được chẩn
  đoán đái tháo đường và có dấu hiệu trên khám/ảnh đáy mắt hoặc chụp mạch. Nếu
  chưa có tiền sử đái tháo đường nhưng nghi ngờ, người bệnh cần làm xét nghiệm.
- ADA 2026 chẩn đoán đái tháo đường bằng HbA1c hoặc glucose huyết tương
  (FPG/OGTT/random glucose trong bối cảnh phù hợp), không dùng ảnh fundus như
  tiêu chuẩn chẩn đoán:
  https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926/2-Diagnosis-and-Classification-of-Diabetes
- Các hệ thống AI võng mạc đã được FDA cho phép như AEYE-DS dùng để phát hiện
  DR ở người lớn **đã được chẩn đoán đái tháo đường**, không dùng để chẩn đoán
  đái tháo đường:
  https://www.accessdata.fda.gov/cdrh_docs/pdf24/K240058.pdf
- Nghiên cứu có thể dự đoán nguy cơ/type 2 diabetes từ ảnh fundus, nhưng đó là
  một bài toán mô hình riêng, cần nhãn đái tháo đường có đối chứng xét nghiệm,
  validation ngoài và ngưỡng vận hành riêng. DR grade 0–4 hiện tại không phải
  nhãn có/không đái tháo đường.

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

- Không gọi “high-risk PDR” từ tổng diện tích hemorrhage. Cần nhận diện tân
  mạch, xuất huyết trước võng mạc/dịch kính và bác sĩ xác nhận.
- Không dùng confidence 70%, diện tích lesion hoặc điểm cộng HbA1c/huyết áp như
  ngưỡng y khoa. Confidence 70% chỉ là cờ vận hành và phải hiệu chỉnh theo bộ
  validation tại cơ sở triển khai.
- Không tự chỉ định anti-VEGF, PRP, laser hoặc phẫu thuật.

## Yêu cầu pháp lý/vận hành Việt Nam

- Luật Khám bệnh, chữa bệnh 15/2023/QH15: quyết định chuyên môn thuộc người
  hành nghề/cơ sở đủ điều kiện.
- Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15, có hiệu lực từ 01/01/2026: khi
  triển khai phải xác định vai trò, căn cứ và trách nhiệm giải trình cho toàn
  bộ vòng đời dữ liệu sức khỏe; đồng thời đối chiếu văn bản hướng dẫn hiện hành.
- Nghị định 98/2021/NĐ-CP, sửa đổi bởi Nghị định 07/2023/NĐ-CP: phải thực hiện
  đánh giá phân loại, hồ sơ và thủ tục thiết bị y tế phù hợp với mục đích sử dụng;
  tài liệu này không tự kết luận hệ thống thuộc loại B hay C.
- Nghị định 13/2023/NĐ-CP là nguồn hồ sơ dự án đang viện dẫn về dữ liệu nhạy
  cảm; phạm vi áp dụng phải được rà soát cùng Luật 91/2025/QH15 và văn bản hướng
  dẫn mới. Tối thiểu cần phân quyền, nhật ký, mã hóa, thời hạn lưu/xóa, đánh giá
  tác động và quy trình sự cố.
- Trước sử dụng thật cần validation đa trung tâm, calibration theo máy/cơ sở,
  giám sát drift, quản lý phiên bản model, quy trình sự cố và bác sĩ nhãn khoa
  phê duyệt toàn bộ rule.
