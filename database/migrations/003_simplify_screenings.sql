BEGIN;

ALTER TABLE screenings
    DROP CONSTRAINT IF EXISTS screenings_left_eye_image_quality_check,
    DROP CONSTRAINT IF EXISTS screenings_right_eye_image_quality_check,
    DROP CONSTRAINT IF EXISTS screenings_review_status_check;

ALTER TABLE screenings
    DROP COLUMN IF EXISTS left_disc_image_url,
    DROP COLUMN IF EXISTS right_disc_image_url,
    DROP COLUMN IF EXISTS left_eye_image_quality,
    DROP COLUMN IF EXISTS right_eye_image_quality,
    DROP COLUMN IF EXISTS review_status,
    DROP COLUMN IF EXISTS clinical_assessment;

COMMIT;
