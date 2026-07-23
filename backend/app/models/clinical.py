from sqlalchemy import (
    Boolean,
    CHAR,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    created_by_account_id = Column(Integer, ForeignKey("accounts.id", ondelete="SET NULL"))
    doctor_id = Column(Integer, ForeignKey("doctors.id", ondelete="SET NULL"))
    screening_date = Column(DateTime(timezone=True), server_default=func.now())
    left_eye_image_url = Column(Text, nullable=True)
    right_eye_image_url = Column(Text, nullable=True)
    status = Column(String(20), default="Pending", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient")
    created_by_account = relationship("Account")
    doctor = relationship("Doctor")
    ai_results = relationship("AIResult", cascade="all, delete-orphan")
    segmentation_results = relationship("LesionSegmentationResult", cascade="all, delete-orphan")
    reviews = relationship("DoctorReview", cascade="all, delete-orphan")


class AIResult(Base):
    __tablename__ = "ai_results"

    id = Column(Integer, primary_key=True, index=True)
    screening_id = Column(Integer, ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False)
    eye = Column(CHAR(1), nullable=False)
    dr_grade = Column(Integer, nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    probabilities = Column(JSONB, nullable=False)
    model_version = Column(String(50), nullable=False)
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("eye IN ('L', 'R')", name="ai_results_eye_check"),
        CheckConstraint("dr_grade BETWEEN 0 AND 4", name="ai_results_dr_grade_check"),
    )


class LesionSegmentationResult(Base):
    __tablename__ = "lesion_segmentation_results"

    id = Column(Integer, primary_key=True, index=True)
    screening_id = Column(Integer, ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False)
    eye = Column(CHAR(1), nullable=False)
    lesion_mask_url = Column(Text)
    microaneurysm_detected = Column(Boolean, default=False)
    microaneurysm_area_pct = Column(Numeric(5, 4), default=0)
    hemorrhage_detected = Column(Boolean, default=False)
    hemorrhage_area_pct = Column(Numeric(5, 4), default=0)
    hard_exudate_detected = Column(Boolean, default=False)
    hard_exudate_area_pct = Column(Numeric(5, 4), default=0)
    dice_score = Column(Numeric(5, 4))
    model_version = Column(String(50), nullable=False)
    segmented_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (CheckConstraint("eye IN ('L', 'R')", name="segmentation_eye_check"),)


class DoctorReview(Base):
    __tablename__ = "doctor_reviews"

    id = Column(Integer, primary_key=True, index=True)
    screening_id = Column(Integer, ForeignKey("screenings.id", ondelete="CASCADE"), nullable=False)
    reviewed_by_account_id = Column(Integer, ForeignKey("accounts.id", ondelete="SET NULL"))
    doctor_id = Column(Integer, ForeignKey("doctors.id", ondelete="SET NULL"))
    eye = Column(CHAR(1), nullable=False)
    final_dr_grade = Column(Integer, nullable=False)
    is_agree_with_ai = Column(Boolean, nullable=False)
    clinical_notes = Column(Text)
    confirmed_at = Column(DateTime(timezone=True), server_default=func.now())

    reviewed_by_account = relationship("Account")
    doctor = relationship("Doctor")

    __table_args__ = (
        CheckConstraint("eye IN ('L', 'R')", name="doctor_reviews_eye_check"),
        CheckConstraint("final_dr_grade BETWEEN 0 AND 4", name="doctor_reviews_grade_check"),
    )


class Recall(Base):
    __tablename__ = "recalls"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    screening_id = Column(Integer, ForeignKey("screenings.id", ondelete="SET NULL"))
    recall_date = Column(Date, nullable=False)
    risk_stratification = Column(String(20), nullable=False)
    recommendation = Column(Text)
    status = Column(String(20), default="Scheduled")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
