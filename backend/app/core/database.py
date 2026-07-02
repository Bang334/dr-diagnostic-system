from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Khởi tạo SQLAlchemy Engine
engine = create_engine(
    settings.DATABASE_URL,
    # pool_pre_ping=True giúp tự động kiểm tra và hồi phục kết nối lỗi
    pool_pre_ping=True
)

# Khởi tạo Session Local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class cho các DB model
Base = declarative_base()

# Dependency để lấy DB Session trong API Routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
