-- A screening may contain the left eye, the right eye, or both.
ALTER TABLE screenings ALTER COLUMN left_eye_image_url DROP NOT NULL;
ALTER TABLE screenings ALTER COLUMN right_eye_image_url DROP NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'screenings_at_least_one_eye_image_check'
    ) THEN
        ALTER TABLE screenings
            ADD CONSTRAINT screenings_at_least_one_eye_image_check
            CHECK (left_eye_image_url IS NOT NULL OR right_eye_image_url IS NOT NULL)
            NOT VALID;
    END IF;
END $$;

ALTER TABLE screenings
    VALIDATE CONSTRAINT screenings_at_least_one_eye_image_check;
