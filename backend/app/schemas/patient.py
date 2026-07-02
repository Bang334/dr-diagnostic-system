from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional

# Các thuộc tính chung của bệnh nhân
class PatientBase(BaseModel):
    patient_code: str = Field(..., max_length=30, description="Mã bệnh nhân định danh duy nhất")
    full_name: str = Field(..., max_length=100, description="Họ và tên bệnh nhân")
    gender: str = Field(..., description="Giới tính (Nam, Nữ, Khác)")
    date_of_birth: date = Field(..., description="Ngày tháng năm sinh")
    phone_number: Optional[str] = Field(None, max_length=15, description="Số điện thoại")
    address: Optional[str] = Field(None, description="Địa chỉ thường trú")
    diabetes_type: Optional[str] = Field(None, description="Phân loại đái tháo đường")
    diabetes_duration_years: Optional[float] = Field(None, ge=0, description="Số năm mắc đái tháo đường")
    latest_hba1c: Optional[float] = Field(None, ge=0, le=20, description="Chỉ số HbA1c gần nhất (%)")

# Dữ liệu truyền lên khi tạo mới bệnh nhân
class PatientCreate(PatientBase):
    pass

# Dữ liệu truyền lên khi cập nhật thông tin bệnh nhân
class PatientUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    phone_number: Optional[str] = Field(None, max_length=15)
    address: Optional[str] = None
    diabetes_type: Optional[str] = None
    diabetes_duration_years: Optional[float] = Field(None, ge=0)
    latest_hba1c: Optional[float] = Field(None, ge=0, le=20)

# Cấu trúc phản hồi từ API (Response)
class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S%z")
        }
