BEGIN;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS patient_id INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'users_patient_id_fkey'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT users_patient_id_fkey
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'users_patient_id_key'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT users_patient_id_key UNIQUE (patient_id);
    END IF;
END $$;

-- Shared initial password: benhnhan
INSERT INTO users (
    username,
    password_hash,
    full_name,
    role,
    patient_id,
    is_active
)
SELECT
    patient.patient_code,
    '$2b$12$Jj00wEmRyI6hg/usjQH9X.smoFYibyT5QI60lU639K.M1J0zAIATy',
    patient.full_name,
    'patient',
    patient.id,
    TRUE
FROM patients AS patient
WHERE NOT EXISTS (
    SELECT 1 FROM users AS account WHERE account.patient_id = patient.id
)
AND NOT EXISTS (
    SELECT 1
    FROM users AS account
    WHERE lower(account.username) = lower(patient.patient_code)
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM patients AS patient
        LEFT JOIN users AS account ON account.patient_id = patient.id
        WHERE account.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Could not create every patient account because one or more patient codes conflict with an existing username.';
    END IF;
END $$;

COMMIT;

