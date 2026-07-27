from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class AccountResponse(BaseModel):
    id: int
    username: str
    full_name: str
    email: Optional[str] = None
    role: str
    hospital_department: Optional[str] = None
    patient_id: Optional[int] = None
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    success: bool
    token: str
    user: AccountResponse
