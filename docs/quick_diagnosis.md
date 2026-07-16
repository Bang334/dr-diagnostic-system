# Nhận diện nhanh bằng checkpoint RETFound-DINOv2

Backend mặc định đọc model tại `backend/checkpoint-best.pth`. Có thể đổi đường dẫn
bằng biến `DR_MODEL_PATH`; dùng `DR_DEVICE=auto`, `cpu` hoặc `cuda` để chọn thiết bị.
Model được nạp ở request đầu tiên để backend khởi động nhanh hơn.

## Chạy backend và frontend

```powershell
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Mở terminal khác:

```powershell
cd frontend
npm install
npm run dev
```

Mở `http://localhost:5173` và chọn **Nhận diện nhanh không cần đăng nhập**.
Sau khi đăng nhập, chức năng tương tự nằm ở tab **Nhận Diện Một Ảnh**.

## Gọi API trực tiếp

```powershell
curl.exe -X POST http://localhost:8000/api/v1/diagnosis/analyze `
  -F "file=@C:\duong-dan\anh-day-mat.jpg"
```

Các endpoint:

- `POST /api/v1/diagnosis/analyze`: upload một ảnh PNG/JPG/JPEG (tối đa 20 MB).
- `GET /api/v1/diagnosis/model-info`: xem checkpoint, kiến trúc, trạng thái nạp và thiết bị.

Luồng sàng lọc bốn ảnh cũng dùng model cục bộ khi
`AI_GRADING_SERVICE_URL=local`. Khi chưa có model phân đoạn tổn thương, giữ
`AI_SEGMENTATION_SERVICE_URL=disabled`; API sẽ đánh dấu segmentation là
`not_available` thay vì tạo kết quả giả.

Kết quả model chỉ hỗ trợ sàng lọc và phải được bác sĩ nhãn khoa xác nhận.
