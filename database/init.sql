-- Database Schema for Diabetic Retinopathy Diagnostic System (PostgreSQL)
-- Created at: 2026-07-02
-- Author: Antigravity AI Assistant

-- Kích hoạt extension UUID để sinh ID tự động nếu cần (tùy chọn)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. BẢNG NGƯỜI DÙNG / BÁC SĨ (Users)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'doctor', -- 'doctor' (bác sĩ nhãn khoa/nội tiết), 'admin' (quản trị hệ thống)
    hospital_department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. BẢNG BỆNH NHÂN TIỂU ĐƯỜNG (Patients)
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    patient_code VARCHAR(30) UNIQUE NOT NULL, -- Mã định danh bệnh nhân (HIS/EMR ID)
    full_name VARCHAR(100) NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('Nam', 'Nữ', 'Khác')) NOT NULL,
    date_of_birth DATE NOT NULL,
    phone_number VARCHAR(15),
    address TEXT,
    diabetes_type VARCHAR(20) CHECK (diabetes_type IN ('Type 1', 'Type 2', 'LADA', 'Thai kỳ', 'Khác')),
    diabetes_duration_years DECIMAL(4,1), -- Số năm mắc bệnh tiểu đường
    latest_hba1c DECIMAL(4,2), -- Chỉ số HbA1c gần nhất (%) để báo cáo dịch tễ
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. BẢNG LỊCH SỬ KHÁM SÀNG LỌC VÕNG MẠC (Screenings)
CREATE TABLE IF NOT EXISTS screenings (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    screening_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ảnh chụp võng mạc nguyên bản (đường dẫn lưu file trên storage)
    left_eye_image_url TEXT NOT NULL,
    right_eye_image_url TEXT NOT NULL,
    left_disc_image_url TEXT,
    right_disc_image_url TEXT,
    
    -- Đánh giá chất lượng ảnh chụp võng mạc (Quality Control)
    left_eye_image_quality VARCHAR(20) DEFAULT 'ReviewRequired' CHECK (left_eye_image_quality IN ('ReviewRequired', 'Good', 'Fair', 'Poor', 'Rejected')),
    right_eye_image_quality VARCHAR(20) DEFAULT 'ReviewRequired' CHECK (right_eye_image_quality IN ('ReviewRequired', 'Good', 'Fair', 'Poor', 'Rejected')),
    
    -- Trạng thái quy trình
    status VARCHAR(20) DEFAULT 'Pending' CHECK (status IN ('Pending', 'AI_Analyzed', 'Reviewed', 'Archived')),
    review_status VARCHAR(20) DEFAULT 'draft' CHECK (review_status IN ('draft', 'confirmed', 'overridden')),
    clinical_assessment JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. BẢNG KẾT QUẢ PHÂN LOẠI MỨC ĐỘ BỆNH VÕNG MẠC TỪ AI (AI Results - Grading)
CREATE TABLE IF NOT EXISTS ai_results (
    id SERIAL PRIMARY KEY,
    screening_id INTEGER REFERENCES screenings(id) ON DELETE CASCADE,
    eye CHAR(1) CHECK (eye IN ('L', 'R')) NOT NULL, -- L: Left, R: Right
    
    -- Phân loại 5 mức độ theo ICDR
    dr_grade INTEGER NOT NULL CHECK (dr_grade BETWEEN 0 AND 4), -- 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR
    confidence DECIMAL(5,4) NOT NULL, -- Độ tin cậy (0.0000 - 1.0000)
    probabilities JSONB NOT NULL, -- Lưu xác suất chi tiết của 5 lớp dưới dạng JSON: {"No_DR": 0.02, "Mild": 0.05, ...}
    
    model_version VARCHAR(50) NOT NULL,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. BẢNG PHÂN ĐOẠN TỔN THƯƠNG VÕNG MẠC TỪ AI (Lesion Segmentation Results)
CREATE TABLE IF NOT EXISTS lesion_segmentation_results (
    id SERIAL PRIMARY KEY,
    screening_id INTEGER REFERENCES screenings(id) ON DELETE CASCADE,
    eye CHAR(1) CHECK (eye IN ('L', 'R')) NOT NULL,
    
    -- Ảnh overlay vẽ bản đồ tổn thương (đường dẫn lưu file ảnh png transparent)
    lesion_mask_url TEXT, 
    
    -- Thống kê diện tích chi tiết các loại tổn thương chính
    microaneurysm_detected BOOLEAN DEFAULT FALSE,
    microaneurysm_area_pct DECIMAL(5,4) DEFAULT 0.0, -- Tỉ lệ diện tích tổn thương trên toàn võng mạc (%)
    
    hemorrhage_detected BOOLEAN DEFAULT FALSE,
    hemorrhage_area_pct DECIMAL(5,4) DEFAULT 0.0,
    
    hard_exudate_detected BOOLEAN DEFAULT FALSE,
    hard_exudate_area_pct DECIMAL(5,4) DEFAULT 0.0,
    
    dice_score DECIMAL(5,4), -- Dice score nội bộ (để đánh giá nếu có ground-truth)
    model_version VARCHAR(50) NOT NULL,
    segmented_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. BẢNG PHÊ DUYỆT CỦA BÁC SĨ (Doctor Reviews)
CREATE TABLE IF NOT EXISTS doctor_reviews (
    id SERIAL PRIMARY KEY,
    screening_id INTEGER REFERENCES screenings(id) ON DELETE CASCADE,
    doctor_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    eye CHAR(1) CHECK (eye IN ('L', 'R')) NOT NULL,
    
    -- Kết quả chuẩn đoán cuối cùng của bác sĩ
    final_dr_grade INTEGER NOT NULL CHECK (final_dr_grade BETWEEN 0 AND 4),
    
    -- Đánh giá mức độ đồng thuận giữa Bác sĩ và AI (để báo cáo hiệu năng AI)
    is_agree_with_ai BOOLEAN NOT NULL, 
    
    clinical_notes TEXT, -- Ghi chú lâm sàng
    confirmed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. BẢNG LỊCH HẸN TÁI KHÁM ĐỊNH KỲ (Recalls)
CREATE TABLE IF NOT EXISTS recalls (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    screening_id INTEGER REFERENCES screenings(id) ON DELETE SET NULL,
    recall_date DATE NOT NULL, -- Ngày tái khám dự kiến
    risk_stratification VARCHAR(20) NOT NULL CHECK (risk_stratification IN ('Low', 'Medium', 'High', 'Urgent')), -- Phân tầng nguy cơ
    recommendation TEXT, -- Khuyến nghị lâm sàng tự động
    status VARCHAR(20) DEFAULT 'Scheduled' CHECK (status IN ('Scheduled', 'Completed', 'Overdue', 'Cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. TẠO CÁC CHỈ MỤC TỐI ƯU TRUY VẤN (Indexes)
CREATE INDEX idx_patients_code ON patients(patient_code);
CREATE INDEX idx_screenings_patient ON screenings(patient_id);
CREATE INDEX idx_screenings_status ON screenings(status);
CREATE INDEX idx_ai_results_screening ON ai_results(screening_id);
CREATE INDEX idx_segmentation_screening ON lesion_segmentation_results(screening_id);
CREATE INDEX idx_doctor_reviews_screening ON doctor_reviews(screening_id);
CREATE INDEX idx_recalls_patient ON recalls(patient_id);
CREATE INDEX idx_recalls_date ON recalls(recall_date);

-- 9. CHÈN DỮ LIỆU MẪU BAN ĐẦU (Seed Data)
-- Chèn tài khoản Admin/Bác sĩ mặc định (Mật khẩu hash cho: "admin123" và "doctor123")
INSERT INTO users (username, password_hash, full_name, email, role, hospital_department)
VALUES 
('admin', '$2b$12$Kk0oA7g/Z8vNisZqB5k7sOr3h30iE.g1Gk2v/XJ4s9YqXp.1uGhy2', 'Quản Trị Viên Hệ Thống', 'admin@hospital.gov.vn', 'admin', 'Công Nghệ Thông Tin'),
('dr.nguyen', '$2b$12$Kk0oA7g/Z8vNisZqB5k7sOr3h30iE.g1Gk2v/XJ4s9YqXp.1uGhy2', 'TS. BS. Nguyễn Văn An', 'an.nv@hospital.gov.vn', 'doctor', 'Khoa Nhãn Khoa'),
('dr.tran', '$2b$12$Kk0oA7g/Z8vNisZqB5k7sOr3h30iE.g1Gk2v/XJ4s9YqXp.1uGhy2', 'ThS. BS. Trần Thị Bình', 'binh.tt@hospital.gov.vn', 'doctor', 'Khoa Nội Tiết');

-- Chèn dữ liệu bệnh nhân mẫu
INSERT INTO patients (patient_code, full_name, gender, date_of_birth, phone_number, address, diabetes_type, diabetes_duration_years, latest_hba1c)
VALUES
('BN0001', 'Phạm Văn Đồng', 'Nam', '1965-04-12', '0912345678', '12 Láng Hạ, Ba Đình, Hà Nội', 'Type 2', 8.5, 7.20),
('BN0002', 'Lê Thị Mai', 'Nữ', '1978-09-25', '0987654321', '45 Nguyễn Trãi, Thanh Xuân, Hà Nội', 'Type 2', 4.0, 6.50),
('BN0003', 'Nguyễn Tiến Dũng', 'Nam', '1952-11-02', '0904445556', '88 Lê Lợi, Hải Châu, Đà Nẵng', 'Type 1', 15.0, 8.40),
('BN0004', 'Hoàng Ngọc Ánh', 'Nữ', '1989-01-30', '0933221100', '123 Cách Mạng Tháng 8, Quận 3, TP. HCM', 'Thai kỳ', 0.5, 5.80);
