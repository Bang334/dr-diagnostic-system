-- Integrate the validated clinical-analysis draft into an existing database.
ALTER TABLE screenings ADD COLUMN IF NOT EXISTS left_disc_image_url TEXT;
ALTER TABLE screenings ADD COLUMN IF NOT EXISTS right_disc_image_url TEXT;
ALTER TABLE screenings ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) DEFAULT 'draft';
ALTER TABLE screenings ADD COLUMN IF NOT EXISTS clinical_assessment JSONB;

ALTER TABLE screenings DROP CONSTRAINT IF EXISTS screenings_left_eye_image_quality_check;
ALTER TABLE screenings DROP CONSTRAINT IF EXISTS screenings_right_eye_image_quality_check;
ALTER TABLE screenings ADD CONSTRAINT screenings_left_eye_image_quality_check
    CHECK (left_eye_image_quality IN ('ReviewRequired', 'Good', 'Fair', 'Poor', 'Rejected'));
ALTER TABLE screenings ADD CONSTRAINT screenings_right_eye_image_quality_check
    CHECK (right_eye_image_quality IN ('ReviewRequired', 'Good', 'Fair', 'Poor', 'Rejected'));

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'screenings_review_status_check'
    ) THEN
        ALTER TABLE screenings ADD CONSTRAINT screenings_review_status_check
            CHECK (review_status IN ('draft', 'confirmed', 'overridden'));
    END IF;
END $$;
