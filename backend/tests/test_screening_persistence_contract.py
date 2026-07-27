import unittest

from app.models.clinical import AIResult, LesionSegmentationResult, Screening


class ScreeningPersistenceContractTests(unittest.TestCase):
    def test_screening_table_contains_only_session_and_two_fundus_images(self):
        self.assertEqual(
            set(Screening.__table__.columns.keys()),
            {
                "id",
                "patient_id",
                "created_by_account_id",
                "doctor_id",
                "screening_date",
                "left_eye_image_url",
                "right_eye_image_url",
                "status",
                "created_at",
            },
        )

    def test_grade_is_persisted_in_ai_results(self):
        columns = set(AIResult.__table__.columns.keys())
        self.assertTrue(
            {"screening_id", "eye", "dr_grade", "confidence", "probabilities"}
            <= columns
        )

    def test_segmentation_labels_are_persisted_in_segmentation_results(self):
        columns = set(LesionSegmentationResult.__table__.columns.keys())
        self.assertTrue(
            {
                "screening_id",
                "eye",
                "microaneurysm_detected",
                "hemorrhage_detected",
                "hard_exudate_detected",
            }
            <= columns
        )


if __name__ == "__main__":
    unittest.main()
