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
    AI_GRADING_SERVICE_URL: str = os.getenv("AI_GRADING_SERVICE_URL", "local")
    AI_SEGMENTATION_SERVICE_URL: str = os.getenv("AI_SEGMENTATION_SERVICE_URL", "disabled")
    AI_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", 120))

    # Local RETFound-DINOv2 grading checkpoint used by /api/v1/diagnosis/analyze.
    DR_MODEL_PATH: str = os.getenv(
        "DR_MODEL_PATH", str(BASE_DIR / "checkpoint-best.pth")
    )
    DR_DEVICE: str = os.getenv("DR_DEVICE", "auto")
    DR_PREPROCESS_ENHANCE: bool = os.getenv("DR_PREPROCESS_ENHANCE", "0") == "1"
    DR_MAX_UPLOAD_BYTES: int = int(
        os.getenv("DR_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    )

    # ── Lesion Segmentation Model Checkpoints (Attention U-Net MA/HE/EX) ──
    LESION_MA_CHECKPOINT: str = os.getenv(
        "LESION_MA_CHECKPOINT",
        str(BASE_DIR / "checkpoints" / "idrid_MA" / "best-checkpoint-epoch=17-val_dice=0.0305.ckpt")
    )
    LESION_HE_CHECKPOINT: str = os.getenv(
        "LESION_HE_CHECKPOINT",
        str(BASE_DIR / "checkpoints" / "idrid_HE" / "best-checkpoint-epoch=70-val_dice=0.0272.ckpt")
    )
    LESION_EX_CHECKPOINT: str = os.getenv(
        "LESION_EX_CHECKPOINT",
        str(BASE_DIR / "checkpoints" / "idrid_EX" / "best-checkpoint-epoch=64-val_dice=0.0414.ckpt")
    )
    LESION_DEVICE: str = os.getenv("LESION_DEVICE", "auto")

settings = Settings()
