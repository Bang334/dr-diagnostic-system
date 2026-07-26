-- PostgreSQL schema for the diabetic-retinopathy screening system.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Authentication and authorization only. Admin is a role, not a separate profile table.
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'doctor'
        CHECK (role IN ('admin', 'doctor', 'patient')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Domain profile for doctors. An admin only receives this profile if they are also a doctor.
CREATE TABLE IF NOT EXISTS doctors (
    id SERIAL PRIMARY KEY,
    account_id INTEGER UNIQUE NOT NULL REFERENCES accounts(id) ON DELETE RESTRICT,
    full_name VARCHAR(100) NOT NULL,
    license_number VARCHAR(50) UNIQUE,
    hospital_department VARCHAR(100),
    specialization VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    account_id INTEGER UNIQUE NOT NULL REFERENCES accounts(id) ON DELETE RESTRICT,
    patient_code VARCHAR(30) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('Nam', 'Nữ', 'Khác')) NOT NULL,
    date_of_birth DATE NOT NULL,
    phone_number VARCHAR(15),
    address TEXT,
    diabetes_type VARCHAR(20)
        CHECK (diabetes_type IN ('Type 1', 'Type 2', 'LADA', 'Thai kỳ', 'Khác')),
    diabetes_duration_years DECIMAL(4,1),
    latest_hba1c DECIMAL(4,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS screenings (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    created_by_account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    doctor_id INTEGER REFERENCES doctors(id) ON DELETE SET NULL,
    screening_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    left_eye_image_url TEXT,
    right_eye_image_url TEXT,
    status VARCHAR(20) DEFAULT 'Pending'
        CHECK (status IN ('Pending', 'AI_Analyzed', 'Reviewed', 'Archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT screenings_at_least_one_eye_image_check
        CHECK (left_eye_image_url IS NOT NULL OR right_eye_image_url IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS ai_results (
    id SERIAL PRIMARY KEY,
    screening_id INTEGER NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    eye CHAR(1) CHECK (eye IN ('L', 'R')) NOT NULL,
    dr_grade INTEGER NOT NULL CHECK (dr_grade BETWEEN 0 AND 4),
    confidence DECIMAL(5,4) NOT NULL,
    probabilities JSONB NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lesion_segmentation_results (
    id SERIAL PRIMARY KEY,
    screening_id INTEGER NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    eye CHAR(1) CHECK (eye IN ('L', 'R')) NOT NULL,
    lesion_mask_url TEXT,
    microaneurysm_detected BOOLEAN DEFAULT FALSE,
    microaneurysm_area_pct DECIMAL(5,4) DEFAULT 0.0,
    hemorrhage_detected BOOLEAN DEFAULT FALSE,
    hemorrhage_area_pct DECIMAL(5,4) DEFAULT 0.0,
    hard_exudate_detected BOOLEAN DEFAULT FALSE,
    hard_exudate_area_pct DECIMAL(5,4) DEFAULT 0.0,
    dice_score DECIMAL(5,4),
    model_version VARCHAR(50) NOT NULL,
    segmented_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS doctor_reviews (
    id SERIAL PRIMARY KEY,
    screening_id INTEGER NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    reviewed_by_account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL,
    doctor_id INTEGER REFERENCES doctors(id) ON DELETE SET NULL,
    eye CHAR(1) CHECK (eye IN ('L', 'R')) NOT NULL,
    final_dr_grade INTEGER NOT NULL CHECK (final_dr_grade BETWEEN 0 AND 4),
    is_agree_with_ai BOOLEAN NOT NULL,
    clinical_notes TEXT,
    confirmed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recalls (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    screening_id INTEGER REFERENCES screenings(id) ON DELETE SET NULL,
    recall_date DATE NOT NULL,
    risk_stratification VARCHAR(20) NOT NULL
        CHECK (risk_stratification IN ('Low', 'Medium', 'High', 'Urgent')),
    recommendation TEXT,
    status VARCHAR(20) DEFAULT 'Scheduled'
        CHECK (status IN ('Scheduled', 'Completed', 'Overdue', 'Cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. TẠO CÁC CHỈ MỤC TỐI ƯU TRUY VẤN (Indexes)
CREATE INDEX IF NOT EXISTS idx_doctors_account ON doctors(account_id);
CREATE INDEX IF NOT EXISTS idx_patients_code ON patients(patient_code);
CREATE INDEX IF NOT EXISTS idx_patients_account ON patients(account_id);
CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);
CREATE INDEX IF NOT EXISTS idx_screenings_status ON screenings(status);
CREATE INDEX IF NOT EXISTS idx_screenings_created_by_account ON screenings(created_by_account_id);
CREATE INDEX IF NOT EXISTS idx_screenings_doctor ON screenings(doctor_id);
CREATE INDEX IF NOT EXISTS idx_ai_results_screening ON ai_results(screening_id);
CREATE INDEX IF NOT EXISTS idx_segmentation_screening ON lesion_segmentation_results(screening_id);
CREATE INDEX IF NOT EXISTS idx_doctor_reviews_screening ON doctor_reviews(screening_id);
CREATE INDEX IF NOT EXISTS idx_doctor_reviews_reviewed_by_account ON doctor_reviews(reviewed_by_account_id);
CREATE INDEX IF NOT EXISTS idx_doctor_reviews_doctor ON doctor_reviews(doctor_id);
CREATE INDEX IF NOT EXISTS idx_recalls_patient ON recalls(patient_id);
CREATE INDEX IF NOT EXISTS idx_recalls_date ON recalls(recall_date);

-- Demo staff accounts. Existing demo password hashes are retained for compatibility.
INSERT INTO accounts (username, password_hash, display_name, email, role)
VALUES
('admin', '$2b$12$Kk0oA7g/Z8vNisZqB5k7sOr3h30iE.g1Gk2v/XJ4s9YqXp.1uGhy2', 'Quản Trị Viên Hệ Thống', 'admin@hospital.gov.vn', 'admin'),
('dr.nguyen', '$2b$12$Kk0oA7g/Z8vNisZqB5k7sOr3h30iE.g1Gk2v/XJ4s9YqXp.1uGhy2', 'TS. BS. Nguyễn Văn An', 'an.nv@hospital.gov.vn', 'doctor'),
('dr.tran', '$2b$12$Kk0oA7g/Z8vNisZqB5k7sOr3h30iE.g1Gk2v/XJ4s9YqXp.1uGhy2', 'ThS. BS. Trần Thị Bình', 'binh.tt@hospital.gov.vn', 'doctor')
ON CONFLICT (username) DO NOTHING;

INSERT INTO doctors (account_id, full_name, hospital_department, specialization)
SELECT id, display_name, 'Khoa Nhãn Khoa', 'Nhãn khoa'
FROM accounts WHERE username = 'dr.nguyen'
ON CONFLICT DO NOTHING;

INSERT INTO doctors (account_id, full_name, hospital_department, specialization)
SELECT id, display_name, 'Khoa Nội Tiết', 'Nội tiết'
FROM accounts WHERE username = 'dr.tran'
ON CONFLICT DO NOTHING;

-- Patient portal accounts. Initial password: benhnhan.
INSERT INTO accounts (username, password_hash, display_name, role)
VALUES
('BN0001', '$2b$12$Jj00wEmRyI6hg/usjQH9X.smoFYibyT5QI60lU639K.M1J0zAIATy', 'Phạm Văn Đồng', 'patient'),
('BN0002', '$2b$12$Jj00wEmRyI6hg/usjQH9X.smoFYibyT5QI60lU639K.M1J0zAIATy', 'Lê Thị Mai', 'patient'),
('BN0003', '$2b$12$Jj00wEmRyI6hg/usjQH9X.smoFYibyT5QI60lU639K.M1J0zAIATy', 'Nguyễn Tiến Dũng', 'patient'),
('BN0004', '$2b$12$Jj00wEmRyI6hg/usjQH9X.smoFYibyT5QI60lU639K.M1J0zAIATy', 'Hoàng Ngọc Ánh', 'patient')
ON CONFLICT (username) DO NOTHING;

INSERT INTO patients (
    account_id,
    patient_code,
    full_name,
    gender,
    date_of_birth,
    phone_number,
    address,
    diabetes_type,
    diabetes_duration_years,
    latest_hba1c
)
SELECT
    account.id,
    seed.patient_code,
    seed.full_name,
    seed.gender,
    seed.date_of_birth::date,
    seed.phone_number,
    seed.address,
    seed.diabetes_type,
    seed.diabetes_duration_years,
    seed.latest_hba1c
FROM (
    VALUES
    ('BN0001', 'Phạm Văn Đồng', 'Nam', '1965-04-12', '0912345678', '12 Láng Hạ, Ba Đình, Hà Nội', 'Type 2', 8.5, 7.20),
    ('BN0002', 'Lê Thị Mai', 'Nữ', '1978-09-25', '0987654321', '45 Nguyễn Trãi, Thanh Xuân, Hà Nội', 'Type 2', 4.0, 6.50),
    ('BN0003', 'Nguyễn Tiến Dũng', 'Nam', '1952-11-02', '0904445556', '88 Lê Lợi, Hải Châu, Đà Nẵng', 'Type 1', 15.0, 8.40),
    ('BN0004', 'Hoàng Ngọc Ánh', 'Nữ', '1989-01-30', '0933221100', '123 Cách Mạng Tháng 8, Quận 3, TP. HCM', 'Thai kỳ', 0.5, 5.80)
) AS seed(
    patient_code,
    full_name,
    gender,
    date_of_birth,
    phone_number,
    address,
    diabetes_type,
    diabetes_duration_years,
    latest_hba1c
)
JOIN accounts AS account ON account.username = seed.patient_code
ON CONFLICT (patient_code) DO NOTHING;

