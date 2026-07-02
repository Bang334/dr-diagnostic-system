# Phân Hệ API Backend (FastAPI)

Phân hệ backend của hệ thống được xây dựng bằng **FastAPI** và kết nối với cơ sở dữ liệu **PostgreSQL** thông qua SQLAlchemy ORM. Backend đóng vai trò làm cổng tích hợp trung tâm, tiếp nhận yêu cầu từ Frontend ReactJS, chuyển tiếp ảnh võng mạc đến mô hình AI để phân tích, chạy các quy tắc lâm sàng (Clinical Rules) và lưu trữ lịch sử bệnh án.

---

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường Phát Triển

### 1. Yêu cầu chuẩn bị
Đảm bảo bạn đã cài đặt Python phiên bản **3.9+** (khuyến nghị **3.10**).

### 2. Thiết lập môi trường ảo và cài đặt thư viện
Tại thư mục `backend/`, thực hiện các lệnh sau:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux / macOS)
source .venv/bin/activate

# Cập nhật pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
1. Nhân bản tệp cấu hình mẫu:
   ```bash
   cp .env.example .env
   ```
2. Mở file `.env` vừa tạo và chỉnh sửa các tham số kết nối PostgreSQL (`DATABASE_URL`) và các cổng dịch vụ AI cho phù hợp với máy cục bộ của bạn.

---

## 🚀 Khởi Chạy API Server

### Chạy trực tiếp bằng Python
Trong môi trường ảo đã kích hoạt, chạy lệnh:
```bash
uvicorn main:app --reload --port 8000
```
* Tham số `--reload` giúp server tự động tải lại code mỗi khi bạn lưu thay đổi.
* Cổng mặc định là `8000`.

### Xem tài liệu API tự động (Swagger / OpenAPI Docs)
Sau khi khởi chạy backend thành công, bạn mở trình duyệt và truy cập:
* **Swagger UI (Interactive Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc (Alternative Docs):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📂 Tổ Chức Code Trong backend/app/

Tuân thủ nguyên tắc thiết kế sạch (Clean Architecture) và DRY:
* `/core/`: Cấu hình hệ thống (`config.py`), kết nối cơ sở dữ liệu (`database.py`) và bảo mật (JWT).
* `/models/`: Các khai báo bảng dữ liệu (SQLAlchemy) tương ứng với database PostgreSQL.
* `/schemas/`: Các Pydantic Schema để validate dữ liệu đầu vào (Request) và định dạng dữ liệu đầu ra (Response).
* `/api//`: Chứa các file Router định nghĩa endpoint API RESTful, phân tách theo từng module (Ví dụ: `patients.py`, `screenings.py`).
* `/services/`: Nơi xử lý các tác vụ logic nặng hoặc tích hợp bên thứ ba (Ví dụ: sinh báo cáo PDF, gọi API của model Deep Learning).
