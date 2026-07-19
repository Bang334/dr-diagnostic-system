from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(
        Integer,
        ForeignKey("accounts.id", ondelete="RESTRICT"),
        unique=True,
        nullable=False,
        index=True,
    )
    full_name = Column(String(100), nullable=False)
    license_number = Column(String(50), unique=True)
    hospital_department = Column(String(100))
    specialization = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    account = relationship("Account", back_populates="doctor_profile")

