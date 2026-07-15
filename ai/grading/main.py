import os
import cv2
import base64
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Thêm đường dẫn gốc để import các module khác
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from ai.preprocessing.fundus_prep import preprocess_fundus_array
from ai.grading.model_handler import DRModelHandler

app = FastAPI(
    title="Diabetic Retinopathy Grading API (Thành viên 1)",
    description="API chạy mô hình EfficientNet-B3 phân loại cấp độ DR.",
    version="1.0"
)

# Cấu hình CORS để Frontend/Backend khác có thể gọi được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo thư mục chứa model trọng số
WEIGHTS_DIR = os.path.join(project_root, "ai", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
MODEL_PATH = os.environ.get(
    "DR_MODEL_PATH", os.path.join(WEIGHTS_DIR, "dr_grading_model.keras")
)
PREPROCESS_ENHANCE = os.environ.get("DR_PREPROCESS_ENHANCE", "0") == "1"

# Khởi tạo thư mục tạm để lưu ảnh upload
TEMP_DIR = os.path.join(project_root, "ai", "grading", "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Biến global lưu instance của model
dr_model = None

@app.on_event("startup")
async def startup_event():
    """Hàm chạy 1 lần duy nhất khi khởi động server, dùng để load model."""
    global dr_model
    print("=== ĐANG KHỞI ĐỘNG DR GRADING API ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Chưa có file model tại {MODEL_PATH}")
        print("[!] Bạn hãy copy file model thật của bạn đè lên đường dẫn này nhé.")

    dr_model = DRModelHandler(model_path=MODEL_PATH)

@app.get("/")
def root():
    return {"message": "DR Grading API is running. Gửi POST request tới /analyze để phân tích ảnh."}

@app.get("/model-info")
def get_model_info():
    """Trả về thông tin về file model đang được load để dễ dàng kiểm chứng."""
    if not os.path.exists(MODEL_PATH):
        return {"error": "Không tìm thấy file model!"}
    
    file_stat = os.stat(MODEL_PATH)
    import datetime
    last_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    size_mb = round(file_stat.st_size / (1024 * 1024), 2)
    
    return {
        "model_path": MODEL_PATH,
        "last_modified": last_modified,
        "size_MB": size_mb,
        "model_version": dr_model.model_version if dr_model else "Unknown"
    }

@app.post("/analyze")
async def analyze_fundus(file: UploadFile = File(...)):
    """
    Endpoint chính nhận file ảnh võng mạc, tiền xử lý và trả về cấp độ DR.
    """
    if dr_model is None or dr_model.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model chưa được load. Vui lòng kiểm tra lại file dr_grading_model.keras trong thư mục ai/weights."
        )

    # 1. Kiểm tra định dạng ảnh
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload không phải là ảnh hợp lệ.")

    # 2. Lưu file ảnh tạm thời vào đĩa cứng
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        img = cv2.imread(temp_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc hoặc xử lý ảnh. Ảnh có thể bị hỏng.")

        # The same colour-preserving crop/resize is used by ai/grading/train.py.
        preprocessed_img = preprocess_fundus_array(
            img,
            img_size=dr_model.input_size[0],
            enhance=PREPROCESS_ENHANCE,
        )

        # 4. Dự đoán qua Model (Inference)
        result = dr_model.predict(preprocessed_img)
        
        # 5. (Tùy chọn) Mã hóa ảnh đã tiền xử lý thành Base64 để Backend xem trước
        _, buffer = cv2.imencode('.png', preprocessed_img)
        b64_string = base64.b64encode(buffer).decode('utf-8')
        result["preprocessed_preview_b64"] = f"data:image/png;base64,{b64_string}"
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống trong quá trình phân tích: {str(e)}")
        
    finally:
        # Xóa file rác sau khi xử lý xong (giải phóng dung lượng)
        if os.path.exists(temp_path):
            os.remove(temp_path)
