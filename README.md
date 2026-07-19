# Hệ Thống Hỗ Trợ Chẩn Đoán Bệnh Võng Mạc Tiểu Đường Từ Ảnh Võng Mạc Ứng Dụng Học Sâu
> **Diabetic Retinopathy Screening & Diagnostic Support System (DR-Screening)**
>
> Đề tài nghiên cứu và xây dựng ứng dụng hỗ trợ bác sĩ nhãn khoa trong việc phát hiện sớm, phân loại 5 mức độ bệnh lý võng mạc tiểu đường (DR) và phân đoạn chi tiết các vùng tổn thương (Microaneurysm, Hemorrhage, Hard Exudate) từ ảnh chụp đáy mắt (Fundus Image).

---

## 📸 Tổng Quan Luồng Xử Lý Hệ Thống

Hệ thống hoạt động theo một luồng tương tác khép kín giữa Bác sĩ, Backend và các mô hình Học Sâu (Deep Learning):

```mermaid
graph TD
    A[Bác sĩ / KTV] -->|1. Upload Ảnh Võng Mạc| B(Frontend ReactJS)
    B -->|2. Gửi request + Metadata| C(Backend FastAPI)
    C -->|3. Gọi Service Preprocessing| D[AI Preprocessing: Green Channel, CLAHE, Ben Graham]
    D -->|4. Ảnh đã chuẩn hóa| E[AI Engine]
    E -->|5.1 Phân loại 5 mức độ - Grading| F[EfficientNet / ConvNeXt / ResNet]
    E -->|5.2 Phân đoạn tổn thương - Segmentation| G[U-Net & Variants]
    F -->|Kết quả DR Grade & Confidence| H[Clinical Orchestrator]
    G -->|Tọa độ & Tỉ lệ vùng tổn thương| H
    H -->|6. Ưu tiên bác sĩ rà soát & Khuyến nghị dự thảo| I[Clinical Rules Engine]
    I -->|7. Tạo báo cáo chi tiết| C
    C -->|8. Lưu trữ dữ liệu| J[(Database PostgreSQL)]
    C -->|9. Trả về kết quả phân tích| B
    B -->|10. Hiển thị bản đồ tổn thương & PDF Report| A
```

---

## 📂 Cấu Trúc Thư Mục Dự Án

Dự án được tổ chức thành các phân hệ độc lập, giúp các thành viên phát triển song song mà không bị xung đột mã nguồn:

```
dr-diagnostic-system/
├── ai/                     # Phân hệ AI (Thành viên 1 & 2)
│   ├── preprocessing/      # Tiền xử lý ảnh (Green channel, CLAHE, Ben Graham)
│   ├── grading/            # Model phân loại DR 5 mức độ (EfficientNet, ResNet, ConvNeXt)
│   ├── segmentation/       # Model phân đoạn tổn thương (U-Net, Attention U-Net)
│   └── semi_supervised/    # Nghiên cứu Semi-supervised & Few-shot Learning
├── backend/                # Server FastAPI & Tích hợp (Thành viên 3)
│   ├── app/
│   │   ├── api/            # API Endpoints (patients, screenings, statistics, auth)
│   │   ├── core/           # Cấu hình hệ thống, bảo mật, kết nối DB
│   │   ├── models/         # Khai báo cấu trúc bảng cơ sở dữ liệu SQLAlchemy
│   │   ├── schemas/        # Định nghĩa Pydantic validation cho request/response
│   │   └── services/       # Xử lý logic lâm sàng, xuất PDF, gọi mô hình AI
│   └── main.py             # Điểm khởi chạy API Server
├── frontend/               # Giao diện Web ứng dụng ReactJS (Thành viên 3)
│   ├── src/
│   │   ├── components/     # UI components dùng chung (Buttons, Cards, Modals)
│   │   ├── screens/        # Các màn hình chính (Phòng khám, Quản lý bệnh nhân, Thống kê)
│   │   ├── services/       # Client gọi APIs từ Backend
│   │   └── theme/          # Bảng màu thiết kế tập trung (Premium HSL CSS)
│   └── package.json
├── database/               # Cơ sở dữ liệu PostgreSQL
│   └── init.sql            # Script khởi tạo cấu trúc bảng & dữ liệu mẫu ban đầu
├── docs/                   # Tài liệu hướng dẫn & Đặc tả dự án
│   ├── api_contract.md     # Đặc tả API Json trao đổi giữa các thành viên
│   ├── clinical_guidelines.md # Thang điểm ICDR/ETDRS & Tiêu chuẩn lâm sàng Việt Nam
│   └── image_standards.md  # Hướng dẫn chụp ảnh võng mạc & tiêu chuẩn chất lượng ảnh
├── docker-compose.yml      # Cấu hình container chạy nhanh PostgreSQL & Services
└── README.md               # Tài liệu tổng quan (File này)
```

---

## 👥 Phân Chia Nhiệm Vụ Trong Nhóm (Cả 3 thành viên cùng làm Website)

| Thành Viên | Phân Vai AI / Lâm Sàng | Nhiệm Vụ AI Chi Tiết | Vai Trò Phát Triển Website Chung |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | **Phân loại mức độ DR (DR Grading)** | - **Nghiên cứu:** Bài toán DR Grading; Thang phân loại ICDR/ETDRS; Các mô hình (EfficientNet, ResNet, ConvNeXt); Các chỉ số đánh giá (Accuracy, Precision, Recall, F1, QWK).<br>- **Thực hành:** Thu thập dữ liệu (EyePACS, APTOS 2019); Tiền xử lý (Resize, Green Channel, CLAHE, Ben Graham); Huấn luyện mô hình phân loại 5 mức; So sánh chọn mô hình tối ưu; Đóng gói mô hình thành API. | - Tham gia thiết kế và xây dựng giao diện bác sĩ (Frontend) hiển thị mức độ phân loại bệnh và độ tin cậy AI.<br>- Tích hợp API Grading vào luồng xử lý chung của Backend. |
| **Thành viên 2** | **Phân đoạn tổn thương (Lesion Segmentation)** | - **Nghiên cứu:** Các loại tổn thương (Microaneurysm, Hemorrhage, Hard Exudate); Các kiến trúc (U-Net, U-Net++, Attention U-Net); Chỉ số đánh giá (Dice Score, IoU).<br>- **Thực hành:** Thu thập dữ liệu (IDRiD, DDR); Tiền xử lý ảnh phục vụ segmentation; Huấn luyện mô hình phân đoạn; Sinh bản đồ tổn thương (mask) dạng overlay. | - Tham gia thiết kế và xây dựng giao diện hiển thị ảnh võng mạc (Frontend) vẽ đè bản đồ tổn thương.<br>- Tích hợp API Segmentation vào luồng xử lý chung của Backend. |
| **Thành viên 3** | **AI Nâng Cao & Hỗ Trợ Lâm Sàng** | - **Nghiên cứu:** Kỹ thuật Semi-supervised Learning, Few-shot Learning; Tiêu chuẩn ứng dụng AI nhãn khoa tại Việt Nam; Quy trình hỗ trợ chẩn đoán lâm sàng.<br>- **Thực hành:** Thử nghiệm Semi-supervised/Few-shot; Xây dựng module tổng hợp kết quả (phân loại + phân đoạn); Sinh báo cáo tự động (mức độ DR, vùng tổn thương, khuyến nghị điều trị dựa trên hướng dẫn lâm sàng). | - Tham gia thiết kế hệ thống, kiến trúc CSDL PostgreSQL.<br>- Xây dựng Module xuất phiếu kết quả PDF, phân hệ thống kê dịch tễ (Dashboard) và quản trị hệ thống. |

> Trạng thái hiện tại: module lâm sàng, adapter AI, một ảnh fundus cho mỗi mắt
> và human review đã được tích hợp. Bộ hai trường ảnh/mắt và PDF hoàn chỉnh vẫn
> là hạng mục cần hoàn thiện. Semi-supervised/Few-shot vẫn là nghiên cứu chưa
> nghiệm thu trên dữ liệu thật; xem `docs/tv3_integration_status.md`. Mọi kết quả
> lâm sàng là dự thảo, không tự chẩn đoán DME hoặc chỉ định điều trị.

---

## 🚀 Hướng Dẫn Cài Đặt Và Chạy Nhanh

### Yêu Cầu Hệ Thống
* Python 3.9+
* Node.js 18+
* PostgreSQL 14+ hoặc Docker Desktop

### 1. Khởi chạy Database nhanh bằng Docker
Tại thư mục gốc dự án, chạy lệnh:
```bash
docker-compose up -d db
```
Hệ thống sẽ khởi tạo một container PostgreSQL lắng nghe ở cổng `5432`, tự động import cấu trúc bảng từ `database/init.sql`.

### 2. Cài đặt và chạy Backend (FastAPI)
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```
API Swagger Docs sẽ khả dụng tại: http://localhost:8000/docs

### 3. Cài đặt và chạy Frontend (ReactJS)
```bash
cd frontend
npm install
npm run dev
```
Giao diện Web sẽ chạy tại: http://localhost:5173

---

## 🛠️ Quy Trình Làm Việc Với Git (Git Workflow)

Để tránh xung đột code, tất cả thành viên bắt buộc tuân thủ quy tắc sau:

1. **Nhánh chính (develop & main):** Không commit trực tiếp lên `main` hay `develop`.
2. **Tạo nhánh mới:** Khi làm nhiệm vụ nào, tạo nhánh từ `develop`:
   * Tính năng mới: `feat/ten-tinh-nang` (Ví dụ: `feat/preprocessing-clahe`)
   * Sửa lỗi: `fix/ten-bug` (Ví dụ: `fix/db-connection-retry`)
3. **Commit:** Tuân thủ chuẩn Conventional Commits:
   * `feat(ai): add clahe preprocessing to dataset loader`
   * `fix(backend): correct database timezone response`
   * `docs: update api contract for grading output`
4. **Merge Request (MR):** Sau khi làm xong, push lên remote và tạo Merge Request trên GitHub để review trước khi merge vào `develop`.

---

## 📜 Tiêu Chuẩn Lâm Sàng Áp Dụng (ICDR)

Hệ thống phân loại ảnh chụp võng mạc thành 5 mức độ bệnh lý theo thang điểm chuẩn lâm sàng quốc tế ICDR (International Clinical Diabetic Retinopathy):

1. **No apparent DR (Cấp 0):** Không thấy dấu hiệu DR trên ảnh đang đánh giá; không đồng nghĩa không mắc đái tháo đường hoặc loại trừ mọi bệnh võng mạc.
2. **Mild NPDR (Cấp 1):** Chỉ có các vi phình mạch (Microaneurysms).
3. **Moderate NPDR (Cấp 2):** Có vi phình mạch, xuất huyết (Hemorrhages), rỉ dịch (Exudates) nhưng chưa đến mức Severe.
4. **Severe NPDR (Cấp 3):** Xuất huyết nặng (>20 điểm ở cả 4 cung phần tư), tĩnh mạch dạng chuỗi (Beading) ở >= 2 cung phần tư, hoặc IRMA ở >= 1 cung phần tư.
5. **Proliferative DR (Cấp 4):** Tân mạch hóa võng mạc (Neovascularization) hoặc xuất huyết dịch kính/trước võng mạc.
