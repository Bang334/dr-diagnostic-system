import time
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.patients import router as patient_router

# Thiết lập ghi log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dr_backend")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Hệ thống hỗ trợ chẩn đoán và sàng lọc bệnh võng mạc tiểu đường từ ảnh chụp đáy mắt.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cấu hình CORS (Cho phép ReactJS Frontend kết nối)
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    "*"  # Cho phép tất cả trong môi trường phát triển
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware để đo thời gian xử lý request (Performance Logging)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Method: {request.method} Path: {request.url.path} Duration: {process_time:.4f}s")
    return response

# XỬ LÝ LỖI TẬP TRUNG (Global Exception Handler - Quy tắc số 9)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log chi tiết lỗi trên server để debug
    logger.error(f"Global Error Catch: {str(exc)}", exc_info=True)
    
    # Trả về thông báo thân thiện và bảo mật cho Client (không để lộ Stack Trace)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Đã xảy ra lỗi hệ thống nghiêm trọng. Vui lòng liên hệ quản trị viên."
        }
    )

@app.exception_handler(status.HTTP_404_NOT_FOUND)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "success": False,
            "error_code": "RESOURCE_NOT_FOUND",
            "message": f"Yêu cầu '{request.url.path}' không tìm thấy trên máy chủ."
        }
    )

# Đăng ký các API Routers
app.include_router(patient_router, prefix="/api/v1")

# Route kiểm tra trạng thái hoạt động (Health Check)
@app.get("/api/v1/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)
