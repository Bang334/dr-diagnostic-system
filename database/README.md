# Phân Hệ Cơ Sở Dữ Liệu PostgreSQL

Thư mục này chứa kịch bản khởi tạo và cấu trúc cơ sở dữ liệu của **Hệ thống hỗ trợ chẩn đoán bệnh võng mạc tiểu đường**. Hệ thống sử dụng PostgreSQL để quản lý hồ sơ bệnh nhân, lịch sử khám, các kết quả phân tích học sâu của AI, duyệt của bác sĩ và thống kê dịch tễ.

---

## 🛠️ Hướng Dẫn Khởi Tạo Database

### Cách 1: Sử dụng Docker Compose (Khuyến nghị)
Nếu máy bạn đã cài đặt Docker Desktop, chỉ cần chạy lệnh sau tại thư mục gốc của dự án:
```bash
docker-compose up -d db
```
Container PostgreSQL sẽ tự động:
1. Tạo database tên là `dr_screening_db`.
2. Tạo tài khoản người dùng `dr_user` với mật khẩu `dr_password_2026`.
3. Chạy file `init.sql` để tạo toàn bộ cấu trúc bảng và chèn dữ liệu mẫu (Seed Data).

### Cách 2: Khởi tạo thủ công trên PostgreSQL cài trên máy
Nếu bạn sử dụng PostgreSQL cài đặt trực tiếp trên hệ điều hành:

1. Đăng nhập vào PostgreSQL CLI (`psql`) hoặc PGAdmin bằng quyền admin (postgres):
   ```sql
   CREATE DATABASE dr_screening_db;
   CREATE USER dr_user WITH PASSWORD 'dr_password_2026';
   GRANT ALL PRIVILEGES ON DATABASE dr_screening_db TO dr_user;
   ```
2. Thực thi file script `init.sql` để khởi tạo các bảng:
   ```bash
   psql -U dr_user -d dr_screening_db -f database/init.sql
   ```

---

## 📊 Mô Hình Thực Thể Quan Hệ (ERD) Tóm Tắt

* **accounts**: Chỉ quản lý đăng nhập, mật khẩu băm, vai trò và trạng thái tài khoản. Admin là một vai trò, không có bảng riêng.
* **doctors**: Hồ sơ nghiệp vụ bác sĩ, liên kết một-một với `accounts` qua `account_id`.
* **patients**: Hồ sơ bệnh nhân, liên kết một-một với tài khoản cổng bệnh nhân qua `account_id`.
* **screenings**: Phiên khám sàng lọc. `created_by_account_id` lưu người thao tác; `doctor_id` chỉ trỏ đến hồ sơ bác sĩ.
* **ai_results**: Kết quả phân loại mức độ DR tự động của AI (0 -> 4) cho từng mắt.
* **lesion_segmentation_results**: Kết quả phân đoạn tự động các loại tổn thương (phình mạch, xuất huyết, rỉ dịch) cho từng mắt.
* **doctor_reviews**: Kết luận cuối cùng; `reviewed_by_account_id` bảo toàn audit người thao tác và `doctor_id` xác định bác sĩ chuyên môn nếu có.
* **recalls**: Lịch tái khám tự động dựa trên mức độ nghiêm trọng của bệnh.

---

## 🔍 Một Số Câu Truy Vấn Thống Kê Dịch Tễ Hữu Ích

Dưới đây là một số câu lệnh SQL mẫu hỗ trợ làm báo cáo dịch tễ học:

### 1. Phân bố mức độ bệnh lý võng mạc tiểu đường (DR Grade) theo kết quả phê duyệt của bác sĩ
```sql
SELECT 
    final_dr_grade,
    CASE 
        WHEN final_dr_grade = 0 THEN 'No DR'
        WHEN final_dr_grade = 1 THEN 'Mild NPDR'
        WHEN final_dr_grade = 2 THEN 'Moderate NPDR'
        WHEN final_dr_grade = 3 THEN 'Severe NPDR'
        WHEN final_dr_grade = 4 THEN 'Proliferative DR'
    END AS severity_label,
    COUNT(*) as patient_count
FROM doctor_reviews
GROUP BY final_dr_grade
ORDER BY final_dr_grade;
```

### 2. Tỉ lệ đồng thuận giữa Bác sĩ và AI
```sql
SELECT 
    COUNT(*) FILTER (WHERE is_agree_with_ai = TRUE) as agree_count,
    COUNT(*) FILTER (WHERE is_agree_with_ai = FALSE) as disagree_count,
    ROUND((COUNT(*) FILTER (WHERE is_agree_with_ai = TRUE) * 100.0) / COUNT(*), 2) as agreement_rate_percentage
FROM doctor_reviews;
```

### 3. Tỉ lệ mắc DR theo nhóm tuổi bệnh nhân
```sql
SELECT 
    CASE 
        WHEN DATE_PART('year', AGE(p.date_of_birth)) < 40 THEN 'Dưới 40 tuổi'
        WHEN DATE_PART('year', AGE(p.date_of_birth)) BETWEEN 40 AND 60 THEN 'Từ 40 - 60 tuổi'
        ELSE 'Trên 60 tuổi'
    END AS age_group,
    COUNT(DISTINCT p.id) as total_screened,
    COUNT(DISTINCT CASE WHEN dr.final_dr_grade > 0 THEN p.id END) as dr_detected,
    ROUND((COUNT(DISTINCT CASE WHEN dr.final_dr_grade > 0 THEN p.id END) * 100.0) / COUNT(DISTINCT p.id), 2) as prevalence_rate
FROM patients p
JOIN screenings s ON p.id = s.patient_id
JOIN doctor_reviews dr ON s.id = dr.screening_id
GROUP BY age_group;
```
