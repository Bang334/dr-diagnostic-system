# Bộ ảnh đáy mắt và kiểm soát khả năng phân loại

Theo Quyết định 2557/QĐ-BYT, quy trình tích hợp yêu cầu tối thiểu **hai ảnh cho
mỗi mắt**:

1. Ảnh đĩa thị.
2. Ảnh võng mạc hậu cực/hoàng điểm.

Hệ thống phải gắn đúng người, đúng mắt, đúng chỉ định. Người chụp/người đọc phải
có chuyên môn phù hợp theo hướng dẫn; trường hợp ảnh không rõ phải hội chẩn,
chụp lại hoặc chuyển chuyên khoa.

## Hai lớp kiểm tra

### Kiểm tra kỹ thuật tự động

- Giải mã được PNG/JPEG thật, không chỉ kiểm tra phần mở rộng/magic bytes.
- Kích thước tối thiểu của prototype: 512 × 512 pixel. Đây là ngưỡng kỹ thuật
  của model, không phải con số do Quyết định 2557 quy định.
- Loại ảnh rỗng, hỏng, quá tối/quá sáng hoặc gần như không có tương phản.

### Xác nhận gradability bởi con người

Ngay cả khi qua kiểm tra kỹ thuật, trạng thái vẫn là `ReviewRequired`. Người đọc
phải xác nhận:

- Đúng mắt và đủ hai trường ảnh.
- Đĩa thị/hoàng điểm và mạch máu cần thiết quan sát được.
- Đủ nét, chiếu sáng và trường nhìn để phân giai đoạn.
- Không bị che khuất bởi đục môi trường, phản quang hoặc mí/mi.

Không tự gắn `Good`. CLAHE hoặc tăng tương phản không được biến ảnh không phân
loại được thành ảnh đủ điều kiện nếu chưa được người đọc xác nhận.
