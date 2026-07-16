import logging
import time

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.auth import router as auth_router
from app.api.diagnosis import router as diagnosis_router
from app.api.patients import router as patient_router
from app.api.reports import router as report_router
from app.api.reviews import router as review_router
from app.api.screenings import router as screening_router
from app.core.config import BASE_DIR, settings
from app.core.security import get_current_user
from app.models.clinical import User


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dr_backend")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Backend FastAPI for diabetic retinopathy screening and diagnostic support.",
    docs_url="/docs",
    redoc_url="/redoc",
)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = (BASE_DIR / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/uploads/{filename}", include_in_schema=False)
def protected_upload(filename: str, current_user: User = Depends(get_current_user)):
    target = (UPLOAD_DIR / filename).resolve()
    if target.parent != UPLOAD_DIR or not target.is_file():
        return JSONResponse(status_code=404, content={"detail": "Image not found."})
    return FileResponse(target)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info("Method=%s Path=%s Duration=%.4fs", request.method, request.url.path, process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Global error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "A system error occurred. Please contact the administrator.",
        },
    )


app.include_router(auth_router, prefix="/api/v1")
app.include_router(diagnosis_router, prefix="/api/v1")
app.include_router(patient_router, prefix="/api/v1")
app.include_router(screening_router, prefix="/api/v1")
app.include_router(review_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")


@app.get("/api/v1/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)
