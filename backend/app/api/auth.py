from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import create_access_token, get_current_account, verify_password
from app.models.account import Account
from app.schemas.auth import AccountResponse, LoginRequest, LoginResponse


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    account = db.query(Account).filter(Account.username == payload.username).first()
    if not account or not account.is_active or not verify_password(payload.password, account.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username or password is incorrect.",
        )

    token = create_access_token(
        {
            "sub": str(account.id),
            "username": account.username,
            "role": account.role,
        }
    )
    return {"success": True, "token": f"Bearer {token}", "user": account}


@router.get("/me", response_model=AccountResponse)
def me(current_account: Account = Depends(get_current_account)):
    return current_account
