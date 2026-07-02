# Phân Hệ Giao Diện Người Dùng (ReactJS + Vite)

Phân hệ Frontend của hệ thống hỗ trợ chẩn đoán võng mạc tiểu đường được phát triển bằng **ReactJS** sử dụng **Vite** làm công cụ đóng gói (bundler) giúp tăng tốc độ phát triển và biên dịch mã nguồn.

Giao diện được thiết kế theo phong cách y khoa hiện đại (Sleek Modern Slate Theme), tự động tương thích giao diện Sáng/Tối (Light/Dark mode) của hệ thống và tối ưu hóa hiển thị ảnh võng mạc y tế cùng bản đồ tổn thương đè (overlay mask).

---

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường Phát Triển

### 1. Yêu cầu chuẩn bị
Đảm bảo máy bạn đã cài đặt **Node.js** phiên bản **18.0.0** trở lên và công cụ quản lý thư viện **npm** đi kèm.

### 2. Cài đặt các thư viện phụ thuộc
Truy cập vào thư mục `frontend/` và chạy lệnh:
```bash
npm install
```

### 3. Cài đặt biến môi trường
1. Nhân bản tệp cấu hình mẫu:
   ```bash
   cp .env.example .env
   ```
2. Cấu hình địa chỉ cổng Backend API (`VITE_API_URL`) nếu chạy ở server khác cổng mặc định `8000`.

---

## 🚀 Khởi Chạy Giao Diện

### Khởi chạy môi trường phát triển (Development)
Chạy lệnh sau tại thư mục `frontend/`:
```bash
npm run dev
```
Giao diện sẽ được mở tại: [http://localhost:5173](http://localhost:5173)

* **Proxy tự động:** Cấu hình Vite đã được thiết lập sẵn proxy để chuyển tiếp các request bắt đầu bằng `/api` sang backend chạy ở `http://localhost:8000`, giúp tránh lỗi CORS khi phát triển cục bộ.

### Đóng gói phiên bản sản xuất (Production Build)
Khi dự án đã sẵn sàng triển khai thực tế:
```bash
npm run build
```
Thư mục `/dist` được tạo ra chứa các tệp HTML, CSS và JS đã được nén tối ưu. Bạn có thể sử dụng Nginx để phân phối thư mục này (Xem `frontend/nginx.conf` và `frontend/Dockerfile`).

---

## 📂 Cấu Trúc Mã Nguồn

* `src/components/`: Chứa các component dùng chung (Ví dụ: `Button.jsx`, `Modal.jsx`, `PatientCard.jsx`).
* `src/screens/`: Chứa các màn hình giao diện chính (Ví dụ: `Dashboard.jsx`, `ScreeningRoom.jsx`, `PatientList.jsx`).
* `src/services/`: Quản lý các hàm gọi API RESTful kết nối với backend.
* `src/theme/`: Chứa định nghĩa mã màu chung (`colors.js`) giúp đồng bộ thiết kế trong toàn hệ thống.
* `src/index.css`: Cài đặt CSS, phông chữ Inter và Outfit, định nghĩa các CSS Variables phục vụ phối màu.
