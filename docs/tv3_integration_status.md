# Trạng thái tích hợp Thành viên 3

## Đã tích hợp vào hệ thống chính

- `backend/app/clinical`: seam duy nhất tổng hợp grading, segmentation, kiểm tra
  bộ ảnh, rule an toàn và báo cáo.
- Adapter HTTP gọi AI grading/segmentation thật; backend không còn sinh grade,
  mask hoặc khuyến nghị ngẫu nhiên.
- Bốn ảnh đầu vào, bối cảnh thị lực/triệu chứng, kết quả từng mắt, trạng thái
  `draft`, bác sĩ confirm và báo cáo PDF.
- Database lưu bộ ảnh đĩa thị/hậu cực và snapshot assessment để báo cáo không
  tự tính lại rule sau khi model/rule đổi phiên bản.

## Semi-supervised và Few-shot

Mã nghiên cứu đã được chuyển vào `ai/semi_supervised` và hiện mới gồm:

- Pseudo-labeling scaffold.
- ProtoNet demo dùng tensor giả lập.

Không tìm thấy dataset, checkpoint, split, log, confidence interval hoặc bảng
so sánh supervised baseline. Vì vậy **không được ghi là đã thử nghiệm thành
công trên dữ liệu võng mạc**. Chỉ nghiệm thu khi có tối thiểu:

1. Patient-level train/validation/test split, không rò rỉ hai mắt/cùng bệnh nhân.
2. Supervised baseline và cùng preprocessing/model seed.
3. QWK, macro-F1, per-class sensitivity/specificity, calibration và confidence
   interval trên test set giữ kín.
4. Phân tích pseudo-label theo lớp, domain/máy chụp và bác sĩ kiểm tra mẫu.
5. Few-shot evaluation trên episode thật, nhiều seed, so với fine-tuning/linear
   probe; không dùng accuracy từ ảnh giả lập làm kết quả đồ án.

Các script nghiên cứu không được nạp vào luồng inference lâm sàng cho đến khi
đạt validation và model được quản lý phiên bản qua adapter AI.
