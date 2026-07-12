import os
from pathlib import Path
from dotenv import load_dotenv

# Đường dẫn đến thư mục root của backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load file .env nếu tồn tại
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class Settings:
    PROJECT_NAME: str = "Diabetic Retinopathy Screening System"
    PROJECT_VERSION: str = "1.0.0"
    
    ENV: str = os.getenv("ENV", "development")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Cấu hình Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://dr_user:dr_password_2026@localhost:5432/dr_screening_db"
    )
    
    # Cấu hình Bảo mật
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        os.getenv("JWT_SECRET", "yoursecretkeyherechangeitinproduction2026"),
    )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", os.getenv("JWT_EXPIRE_MINUTES", 1440))
    )
    
    # Các cổng dịch vụ AI
    AI_GRADING_SERVICE_URL: str = os.getenv("AI_GRADING_SERVICE_URL", "http://localhost:8001")
    AI_SEGMENTATION_SERVICE_URL: str = os.getenv("AI_SEGMENTATION_SERVICE_URL", "http://localhost:8002")
    AI_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", 120))

settings = Settings()
