# app/api/main.py

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# --- Project Imports ---
from app.api.endpoints import training, prediction # Import các modules chứa router
from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION

# --- Khởi tạo FastAPI App ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    # Có thể thêm các cấu hình khác ở đây nếu cần
    # docs_url="/docs", # Đường dẫn đến Swagger UI (mặc định)
    # redoc_url="/redoc" # Đường dẫn đến ReDoc (mặc định)
)

# --- Gắn các Routers ---
# Bao gồm router cho training endpoints với prefix /training
app.include_router(training.router, prefix="/training", tags=["Training"])
# Bao gồm router cho prediction endpoints với prefix /prediction
app.include_router(prediction.router, prefix="/prediction", tags=["Prediction"])

# --- Endpoint gốc (Optional) ---
@app.get("/", include_in_schema=False)
async def root():
    """Chuyển hướng đến trang tài liệu API (Swagger UI)."""
    return RedirectResponse(url="/docs")

# --- (Optional) Thêm các xử lý khác nếu cần ---
# Ví dụ: Middleware, Exception Handlers tùy chỉnh, ...

# --- Lệnh chạy (thường dùng với uvicorn từ terminal) ---
# Ví dụ: uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000