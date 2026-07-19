import unittest

from pydantic import ValidationError

from app.schemas.review import ReviewCreate


class ReviewContractTests(unittest.TestCase):
    def test_single_eye_screening_can_be_reviewed(self):
        payload = ReviewCreate.model_validate(
            {
                "left_eye_review": {
                    "final_dr_grade": 2,
                    "is_agree_with_ai": True,
                    "clinical_notes": "Theo dõi định kỳ.",
                },
                "recall_settings": {
                    "recall_in_months": 6,
                    "risk_stratification": "Medium",
                },
            }
        )

        self.assertIsNotNone(payload.left_eye_review)
        self.assertIsNone(payload.right_eye_review)

    def test_review_requires_at_least_one_eye(self):
        with self.assertRaises(ValidationError):
            ReviewCreate.model_validate(
                {
                    "recall_settings": {
                        "recall_in_months": 12,
                        "risk_stratification": "Low",
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
