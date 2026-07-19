from sqlalchemy import Column, Integer, String, Date, Numeric, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(
        Integer,
        ForeignKey("accounts.id", ondelete="RESTRICT"),
        unique=True,
        nullable=False,
        index=True,
    )
    patient_code = Column(String(30), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    phone_number = Column(String(15), nullable=True)
    address = Column(Text, nullable=True)
    diabetes_type = Column(String(20), nullable=True)
    diabetes_duration_years = Column(Numeric(4, 1), nullable=True)
    latest_hba1c = Column(Numeric(4, 2), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    account = relationship("Account", back_populates="patient_profile", uselist=False)

    @property
    def portal_username(self):
        return self.account.username if self.account else None

    @property
    def has_portal_account(self):
        return self.account is not None

    def __repr__(self):
        return f"<Patient(code={self.patient_code}, name={self.full_name})>"
