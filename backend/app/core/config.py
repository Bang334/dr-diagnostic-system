import os
from pathlib import Path
from dotenv import load_dotenv

# Đường dẫn đến thư mục root của backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Thư mục tập trung lưu model đã train (nằm ở project root, ngoài backend/)
WEIGHTS_DIR = BASE_DIR.parent / "ai" / "weights"

# Load file .env nếu tồn tại
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable from a list of aliases."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _backend_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


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
    PATIENT_DEFAULT_PASSWORD: str = os.getenv("PATIENT_DEFAULT_PASSWORD", "benhnhan")

    # Các cổng dịch vụ AI
    AI_GRADING_SERVICE_URL: str = os.getenv("AI_GRADING_SERVICE_URL", "local")
    AI_SEGMENTATION_SERVICE_URL: str = os.getenv("AI_SEGMENTATION_SERVICE_URL", "disabled")
    AI_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", 120))

    # Gemini writes a de-identified clinical draft after local image analysis.
    # The key must remain server-side and must never use a VITE_ prefix.
    GEMINI_API_KEY: str = _first_env("GEMINI_API_KEY", "GEMINI_API_KEYS")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    GEMINI_TIMEOUT_SECONDS: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", 45))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", 0.2))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", 4096))

    # Cloud storage for sanitized fundus images. Credentials stay server-side.
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "").strip()
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "").strip()
    CLOUDINARY_FOLDER: str = os.getenv(
        "CLOUDINARY_FOLDER", "dr-diagnostic-system/fundus"
    ).strip()

    # Local RETFound-DINOv2 grading checkpoint used by the screening workflow.
    DR_MODEL_PATH: str = _backend_path(
        os.getenv("DR_MODEL_PATH", str(WEIGHTS_DIR / "grading" / "checkpoint-best.pth"))
    )
    DR_FEWSHOT_MODEL_PATH: str = _backend_path(
        os.getenv("DR_FEWSHOT_MODEL_PATH", str(WEIGHTS_DIR / "grading" / "best-fewshot.pth"))
    )
    DR_DEFAULT_MODEL: str = os.getenv("DR_DEFAULT_MODEL", "grading")
    DR_DEVICE: str = os.getenv("DR_DEVICE", "auto")
    DR_PREPROCESS_ENHANCE: bool = os.getenv("DR_PREPROCESS_ENHANCE", "0") == "1"
    DR_MAX_UPLOAD_BYTES: int = int(
        os.getenv("DR_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    )

    # ── Lesion Segmentation Model Checkpoints (Attention U-Net MA/HE/EX) ──
    LESION_MA_CHECKPOINT: str = os.getenv(
        "LESION_MA_CHECKPOINT",
        str(WEIGHTS_DIR / "segmentation" / "idrid_MA" / "best-checkpoint-epoch=17-val_dice=0.0305.ckpt")
    )
    LESION_HE_CHECKPOINT: str = os.getenv(
        "LESION_HE_CHECKPOINT",
        str(WEIGHTS_DIR / "segmentation" / "idrid_HE" / "best-checkpoint-epoch=70-val_dice=0.0272.ckpt")
    )
    LESION_EX_CHECKPOINT: str = os.getenv(
        "LESION_EX_CHECKPOINT",
        str(WEIGHTS_DIR / "segmentation" / "idrid_EX" / "best-checkpoint-epoch=64-val_dice=0.0414.ckpt")
    )
    LESION_DEVICE: str = os.getenv("LESION_DEVICE", "auto")
    SEGMENTATION_LESION_PATH: str = os.getenv("SEGMENTATION_LESION_PATH", "ai/segmentation")


settings = Settings()
