from sqlalchemy import Boolean, Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    role = Column(String(20), nullable=False, default="doctor")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    doctor_profile = relationship("Doctor", back_populates="account", uselist=False)
    patient_profile = relationship("Patient", back_populates="account", uselist=False)

    @property
    def full_name(self):
        if self.role == "doctor" and self.doctor_profile:
            return self.doctor_profile.full_name
        if self.role == "patient" and self.patient_profile:
            return self.patient_profile.full_name
        return self.display_name

    @property
    def hospital_department(self):
        return self.doctor_profile.hospital_department if self.doctor_profile else None

    @property
    def patient_id(self):
        return self.patient_profile.id if self.patient_profile else None

