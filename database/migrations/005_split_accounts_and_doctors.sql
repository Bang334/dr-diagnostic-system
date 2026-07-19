BEGIN;

LOCK TABLE users, patients, screenings, doctor_reviews IN ACCESS EXCLUSIVE MODE;

ALTER TABLE users RENAME TO accounts;
ALTER TABLE accounts RENAME COLUMN full_name TO display_name;

DO $$
BEGIN
    IF to_regclass('public.users_id_seq') IS NOT NULL
       AND to_regclass('public.accounts_id_seq') IS NULL THEN
        ALTER SEQUENCE users_id_seq RENAME TO accounts_id_seq;
    END IF;
END $$;

CREATE TABLE doctors (
    id SERIAL PRIMARY KEY,
    account_id INTEGER UNIQUE NOT NULL
        REFERENCES accounts(id) ON DELETE RESTRICT,
    full_name VARCHAR(100) NOT NULL,
    license_number VARCHAR(50) UNIQUE,
    hospital_department VARCHAR(100),
    specialization VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO doctors (
    id,
    account_id,
    full_name,
    hospital_department,
    specialization
)
SELECT
    id,
    id,
    display_name,
    hospital_department,
    CASE
        WHEN lower(coalesce(hospital_department, '')) LIKE '%nhãn%' THEN 'Nhãn khoa'
        WHEN lower(coalesce(hospital_department, '')) LIKE '%nội tiết%' THEN 'Nội tiết'
        ELSE NULL
    END
FROM accounts
WHERE role = 'doctor';

SELECT setval(
    pg_get_serial_sequence('doctors', 'id'),
    GREATEST(COALESCE((SELECT MAX(id) FROM doctors), 1), 1),
    TRUE
);

ALTER TABLE patients ADD COLUMN account_id INTEGER;

UPDATE patients AS patient
SET account_id = account.id
FROM accounts AS account
WHERE account.patient_id = patient.id;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM patients WHERE account_id IS NULL) THEN
        RAISE EXCEPTION 'Every patient must have a linked account before migration.';
    END IF;
END $$;

ALTER TABLE patients
    ALTER COLUMN account_id SET NOT NULL,
    ADD CONSTRAINT patients_account_id_key UNIQUE (account_id),
    ADD CONSTRAINT patients_account_id_fkey
        FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE RESTRICT;

ALTER TABLE screenings RENAME COLUMN doctor_id TO created_by_account_id;
ALTER TABLE screenings
    DROP CONSTRAINT IF EXISTS screenings_doctor_id_fkey,
    ADD CONSTRAINT screenings_created_by_account_id_fkey
        FOREIGN KEY (created_by_account_id) REFERENCES accounts(id) ON DELETE SET NULL,
    ADD COLUMN doctor_id INTEGER;

UPDATE screenings AS screening
SET doctor_id = doctor.id
FROM doctors AS doctor
WHERE doctor.account_id = screening.created_by_account_id;

ALTER TABLE screenings
    ADD CONSTRAINT screenings_doctor_id_fkey
        FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE SET NULL;

ALTER TABLE doctor_reviews RENAME COLUMN doctor_id TO reviewed_by_account_id;
ALTER TABLE doctor_reviews
    DROP CONSTRAINT IF EXISTS doctor_reviews_doctor_id_fkey,
    ADD CONSTRAINT doctor_reviews_reviewed_by_account_id_fkey
        FOREIGN KEY (reviewed_by_account_id) REFERENCES accounts(id) ON DELETE SET NULL,
    ADD COLUMN doctor_id INTEGER;

UPDATE doctor_reviews AS review
SET doctor_id = doctor.id
FROM doctors AS doctor
WHERE doctor.account_id = review.reviewed_by_account_id;

ALTER TABLE doctor_reviews
    ADD CONSTRAINT doctor_reviews_doctor_id_fkey
        FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE SET NULL;

-- If an admin created a screening but a doctor later reviewed it, assign that doctor.
UPDATE screenings AS screening
SET doctor_id = reviewer.doctor_id
FROM (
    SELECT screening_id, MIN(doctor_id) AS doctor_id
    FROM doctor_reviews
    WHERE doctor_id IS NOT NULL
    GROUP BY screening_id
) AS reviewer
WHERE screening.id = reviewer.screening_id
  AND screening.doctor_id IS NULL;

ALTER TABLE accounts
    DROP COLUMN patient_id,
    DROP COLUMN hospital_department;

CREATE INDEX idx_doctors_account ON doctors(account_id);
CREATE INDEX idx_patients_account ON patients(account_id);
CREATE INDEX idx_screenings_created_by_account ON screenings(created_by_account_id);
CREATE INDEX idx_screenings_doctor ON screenings(doctor_id);
CREATE INDEX idx_doctor_reviews_reviewed_by_account ON doctor_reviews(reviewed_by_account_id);
CREATE INDEX idx_doctor_reviews_doctor ON doctor_reviews(doctor_id);

COMMIT;

