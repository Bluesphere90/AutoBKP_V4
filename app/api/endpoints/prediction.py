# app/api/endpoints/prediction.py

import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Body, Path as FastApiPath
from typing import List

# --- Project Imports ---
from app.api import schemas # Import các Pydantic models đã cập nhật
import app.core.config as config # Import config module
# Import các hàm dự đoán đã refactor VÀ hàm mới
from app.ml.models import (
    predict_combined,
    predict_hachtoan_only,
    predict_mahanghoa_only,
    predict_mahanghoa_direct # <-- Hàm mới
)
from app.ml.data_handler import prepare_prediction_data
from app.ml.utils import get_client_models_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependencies kiểm tra model ---
# Dependency kiểm tra HachToan components (Giữ nguyên)
async def check_hachtoan_model(client_id: str = FastApiPath(...)):
    models_path = get_client_models_path(client_id)
    required_ht_preprocessor = models_path / config.PREPROCESSOR_HACHTOAN_FILENAME
    required_ht_encoder = models_path / "label_encoders" / config.HACHTOAN_ENCODER_FILENAME
    if not required_ht_preprocessor.exists() or not required_ht_encoder.exists():
        raise HTTPException(status_code=404, detail=f"Mô hình HachToan chưa sẵn sàng cho client '{client_id}'.")
    logger.debug(f"Client {client_id}: Thành phần model HachToan OK.")

# Dependency kiểm tra MaHangHoa (Dependent) components (Giữ nguyên)
async def check_mahanghoa_model(client_id: str = FastApiPath(...)):
    models_path = get_client_models_path(client_id)
    required_mh_preprocessor = models_path / config.PREPROCESSOR_MAHANGHOA_FILENAME
    required_mh_encoder = models_path / "label_encoders" / config.MAHANGHOA_ENCODER_FILENAME
    # Thêm kiểm tra model MH nếu muốn chặt chẽ hơn
    required_mh_model = models_path / config.MAHANGHOA_MODEL_FILENAME
    if not required_mh_preprocessor.exists() or not required_mh_encoder.exists() or not required_mh_model.exists():
        raise HTTPException(status_code=404, detail=f"Mô hình MaHangHoa (dependent) chưa sẵn sàng cho client '{client_id}'.")
    logger.debug(f"Client {client_id}: Thành phần model MaHangHoa (dependent) OK.")

# --- Dependency MỚI kiểm tra MaHangHoa Direct components ---
async def check_mahanghoa_direct_model(client_id: str = FastApiPath(...)):
    """Kiểm tra preprocessor HT, encoder MH, và model MH Direct."""
    models_path = get_client_models_path(client_id)
    required_ht_preprocessor = models_path / config.PREPROCESSOR_HACHTOAN_FILENAME # Dùng prep HT
    required_mh_encoder = models_path / "label_encoders" / config.MAHANGHOA_ENCODER_FILENAME # Dùng enc MH
    required_mh_direct_model = models_path / config.MAHANGHOA_DIRECT_MODEL_FILENAME # Model mới
    if not required_ht_preprocessor.exists() or not required_mh_encoder.exists() or not required_mh_direct_model.exists():
        raise HTTPException(status_code=404, detail=f"Mô hình MaHangHoa Direct chưa sẵn sàng cho client '{client_id}'.")
    logger.debug(f"Client {client_id}: Thành phần model MaHangHoa Direct OK.")


# --- API Endpoints ---

# Endpoint Gốc (Dự đoán kết hợp - Model 1 -> Model 2)
@router.post(
    "/{client_id}",
    response_model=schemas.PredictionResponse,
    summary="Dự đoán kết hợp HachToan và MaHangHoa",
    dependencies=[Depends(check_hachtoan_model)], # Chỉ cần check HT ban đầu
    tags=["Prediction"],
    responses={...} # Giữ nguyên responses
)
async def predict_combined_endpoint(
    client_id: str = FastApiPath(..., description="ID khách hàng"),
    request_body: schemas.PredictionRequest = Body(...)
) -> schemas.PredictionResponse:
    # ... (Logic gọi predict_combined giữ nguyên như trước) ...
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán KẾT HỢP.")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
        if input_df.empty and request_body.items: raise ValueError("Input rỗng sau chuẩn bị.")
    except Exception as e: raise HTTPException(status_code=400, detail=f"Lỗi xử lý input: {e}")
    try:
        prediction_results = predict_combined(client_id, input_df)
        if len(prediction_results) != len(request_body.items): raise HTTPException(status_code=500, detail="Lỗi số lượng kết quả.")
        response_items = [schemas.PredictionResultItem(**result) for result in prediction_results]
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi predict_combined: {e}", exc_info=True)
        error_items = [schemas.PredictionResultItem(is_outlier_input1=False, is_outlier_input2=False, error=f"Lỗi nội bộ: {e}") for _ in request_body.items]
        return schemas.PredictionResponse(results=error_items)
    return schemas.PredictionResponse(results=response_items)


# Endpoint Chỉ dự đoán HachToan (Model 1)
@router.post(
    "/{client_id}/hachtoan",
    response_model=schemas.HachToanPredictionResponse,
    summary="Chỉ dự đoán HachToan",
    dependencies=[Depends(check_hachtoan_model)],
    tags=["Prediction"],
    responses={...} # Giữ nguyên
)
async def predict_hachtoan_endpoint(
    client_id: str = FastApiPath(..., description="ID khách hàng"),
    request_body: schemas.PredictionRequest = Body(...)
) -> schemas.HachToanPredictionResponse:
    # ... (Logic gọi predict_hachtoan_only giữ nguyên như trước) ...
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán CHỈ HACHTOAN.")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
        if input_df.empty and request_body.items: raise ValueError("Input rỗng sau chuẩn bị.")
    except Exception as e: raise HTTPException(status_code=400, detail=f"Lỗi xử lý input: {e}")
    try:
        prediction_results = predict_hachtoan_only(client_id, input_df)
        if len(prediction_results) != len(request_body.items): raise HTTPException(status_code=500, detail="Lỗi số lượng kết quả.")
        response_items = [schemas.HachToanPredictionResultItem(**result) for result in prediction_results]
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi predict_hachtoan_only: {e}", exc_info=True)
        error_items = [schemas.HachToanPredictionResultItem(is_outlier_input1=False, error=f"Lỗi nội bộ: {e}") for _ in request_body.items]
        return schemas.HachToanPredictionResponse(results=error_items)
    return schemas.HachToanPredictionResponse(results=response_items)


# Endpoint Chỉ dự đoán MaHangHoa (Dependent - Model 2)
@router.post(
    "/{client_id}/mahanghoa",
    response_model=schemas.MaHangHoaPredictionResponse,
    summary="Chỉ dự đoán MaHangHoa (cần cung cấp HachToan)",
    dependencies=[Depends(check_mahanghoa_model)], # Check model MH dependent
    tags=["Prediction"],
    responses={...} # Giữ nguyên
)
async def predict_mahanghoa_endpoint(
    client_id: str = FastApiPath(..., description="ID khách hàng"),
    request_body: schemas.MaHangHoaPredictionRequest = Body(...) # Dùng request schema có HachToan
) -> schemas.MaHangHoaPredictionResponse:
    # ... (Logic gọi predict_mahanghoa_only giữ nguyên như trước) ...
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán CHỈ MAHANGHOA (dependent).")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
        if input_df.empty and request_body.items: raise ValueError("Input rỗng sau chuẩn bị.")
        if config.TARGET_HACHTOAN not in input_df.columns: raise ValueError(f"Thiếu cột {config.TARGET_HACHTOAN}.")
    except Exception as e: raise HTTPException(status_code=400, detail=f"Lỗi xử lý input: {e}")
    try:
        prediction_results = predict_mahanghoa_only(client_id, input_df)
        if len(prediction_results) != len(request_body.items): raise HTTPException(status_code=500, detail="Lỗi số lượng kết quả.")
        response_items = [schemas.MaHangHoaPredictionResultItem(**result) for result in prediction_results]
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi predict_mahanghoa_only: {e}", exc_info=True)
        error_items = [schemas.MaHangHoaPredictionResultItem(is_outlier_input2=False, error=f"Lỗi nội bộ: {e}") for _ in request_body.items]
        return schemas.MaHangHoaPredictionResponse(results=error_items)
    return schemas.MaHangHoaPredictionResponse(results=response_items)


# --- Endpoint MỚI: Chỉ dự đoán MaHangHoa (Direct - Model 3) ---
@router.post(
    "/{client_id}/mahanghoa/direct", # Đường dẫn mới
    response_model=schemas.MaHangHoaDirectPredictionResponse, # Dùng response schema mới
    summary="Chỉ dự đoán MaHangHoa (trực tiếp từ input gốc)",
    description="Nhận input cơ bản (MSTNguoiBan, TenHangHoaDichVu) và dự đoán trực tiếp MaHangHoa.",
    dependencies=[Depends(check_mahanghoa_direct_model)], # Check model MH Direct
    tags=["Prediction"],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Không tìm thấy mô hình MaHangHoa Direct cho client"},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input"},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
    }
)
async def predict_mahanghoa_direct_endpoint( # Tên hàm endpoint mới
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    request_body: schemas.PredictionRequest = Body(...) # Dùng request schema gốc (chỉ cần input cơ bản)
) -> schemas.MaHangHoaDirectPredictionResponse:
    """Endpoint chỉ dự đoán MaHangHoa trực tiếp."""
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán CHỈ MAHANGHOA (direct) cho {len(request_body.items)} bản ghi.")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
        if input_df.empty and request_body.items: raise ValueError("Input rỗng sau chuẩn bị.")
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi chuẩn bị dữ liệu (MH direct): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu input: {e}")

    try:
        # Gọi hàm dự đoán MH Direct mới
        prediction_results: List[dict] = predict_mahanghoa_direct(client_id, input_df)

        if len(prediction_results) != len(request_body.items):
             logger.error(f"Client {client_id}: Số lượng kết quả (MH direct) không khớp input.")
             raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        # Chuyển đổi sang schema response mới
        response_items = [schemas.MaHangHoaDirectPredictionResultItem(**result) for result in prediction_results]

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi gọi predict_mahanghoa_direct: {e}", exc_info=True)
        error_items = [schemas.MaHangHoaDirectPredictionResultItem(is_outlier_input1=False, error=f"Lỗi dự đoán nội bộ: {e}") for _ in request_body.items]
        return schemas.MaHangHoaDirectPredictionResponse(results=error_items)

    logger.info(f"Client {client_id}: Dự đoán chỉ MaHangHoa (direct) hoàn tất.")
    return schemas.MaHangHoaDirectPredictionResponse(results=response_items)