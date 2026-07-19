import unittest

from app.models.account import Account
from app.models.clinical import DoctorReview, Screening
from app.models.doctor import Doctor
from app.models.patient import Patient


def foreign_key_targets(model, column_name):
    column = model.__table__.columns[column_name]
    return {foreign_key.target_fullname for foreign_key in column.foreign_keys}


class AccountDomainModelTests(unittest.TestCase):
    def test_account_contains_only_shared_identity_fields(self):
        columns = set(Account.__table__.columns.keys())
        self.assertEqual(Account.__tablename__, "accounts")
        self.assertNotIn("patient_id", columns)
        self.assertNotIn("hospital_department", columns)
        self.assertTrue(
            {"username", "password_hash", "display_name", "role", "is_active"}
            <= columns
        )

    def test_doctor_and_patient_profiles_reference_accounts(self):
        self.assertEqual(foreign_key_targets(Doctor, "account_id"), {"accounts.id"})
        self.assertEqual(foreign_key_targets(Patient, "account_id"), {"accounts.id"})

    def test_clinical_records_separate_actor_audit_from_doctor_profile(self):
        self.assertEqual(
            foreign_key_targets(Screening, "created_by_account_id"),
            {"accounts.id"},
        )
        self.assertEqual(foreign_key_targets(Screening, "doctor_id"), {"doctors.id"})
        self.assertEqual(
            foreign_key_targets(DoctorReview, "reviewed_by_account_id"),
            {"accounts.id"},
        )
        self.assertEqual(
            foreign_key_targets(DoctorReview, "doctor_id"),
            {"doctors.id"},
        )


if __name__ == "__main__":
    unittest.main()

