# app/api/endpoints/column_config.py

import logging
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Body, Path as FastApiPath

# --- Project Imports ---
from app.api import schemas # Import các Pydantic models đã cập nhật
from app.ml.utils import get_client_models_path # Để lấy đường dẫn lưu config
import app.core.config as config # Import config để dùng tên file

# --- Setup Logger and Router ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
CONFIG_FILENAME = "column_config.json" # Tên file cấu hình cố định

# --- API Endpoints ---

@router.put(
    "/columns/{client_id}",
    response_model=schemas.MessageResponse,
    summary="Tạo hoặc cập nhật cấu hình cột cho client",
    description=f"Ghi đè hoàn toàn file '{CONFIG_FILENAME}' bằng nội dung JSON được cung cấp.",
    tags=["Configuration"], # Tag mới
    status_code=200, # OK hoặc 201 Created nếu muốn phân biệt
    responses={
        400: {"model": schemas.ErrorResponse, "description": "Dữ liệu cấu hình không hợp lệ."},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input."},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server khi lưu file."}
    }
)
async def set_column_config(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    config_data: schemas.ColumnConfigRequest = Body(..., description="Nội dung JSON của cấu hình cột.")
):
    """
    Endpoint để đặt (tạo mới hoặc ghi đè) file cấu hình cột cho client.
    """
    logger.info(f"Nhận yêu cầu cập nhật cấu hình cột cho client: {client_id}")
    models_path = get_client_models_path(client_id) # Hàm này đã tạo thư mục nếu chưa có
    config_file_path = models_path / CONFIG_FILENAME

    try:
        # Pydantic đã validate cấu trúc cơ bản.
        # Chuyển Pydantic model thành dict để lưu JSON
        config_dict = config_data.model_dump(mode='json') # Dùng mode='json' để xử lý Enum

        # Ghi file JSON
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)

        logger.info(f"Đã lưu/cập nhật thành công file cấu hình cột: {config_file_path}")
        return schemas.MessageResponse(
            message=f"Đã cập nhật thành công cấu hình cột cho client '{client_id}'.",
            client_id=client_id
        )
    except Exception as e:
        logger.error(f"Lỗi khi lưu file cấu hình cột cho client {client_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi server khi lưu file cấu hình: {e}")


@router.get(
    "/columns/{client_id}",
    response_model=schemas.ColumnConfigResponse, # Dùng lại schema request làm response
    summary="Lấy cấu hình cột hiện tại của client",
    description=f"Đọc và trả về nội dung file '{CONFIG_FILENAME}' nếu tồn tại.",
    tags=["Configuration"],
    responses={
        404: {"model": schemas.ErrorResponse, "description": f"Không tìm thấy file cấu hình '{CONFIG_FILENAME}' cho client."},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server khi đọc file."}
    }
)
async def get_column_config(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc")
):
    """
    Endpoint để lấy file cấu hình cột hiện tại của client.
    """
    logger.debug(f"Nhận yêu cầu lấy cấu hình cột cho client: {client_id}")
    models_path = get_client_models_path(client_id)
    config_file_path = models_path / CONFIG_FILENAME

    if not config_file_path.exists():
        logger.warning(f"File cấu hình cột không tồn tại cho client {client_id}: {config_file_path}")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file cấu hình cột cho client '{client_id}'.")

    try:
        # Đọc file JSON
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Có thể validate lại dữ liệu đọc được bằng schema nếu muốn chắc chắn
        # validated_data = schemas.ColumnConfigRequest(**config_data)
        # return validated_data
        logger.debug(f"Đã đọc thành công file cấu hình cột cho client {client_id}")
        return config_data # Trả về dict đã đọc

    except json.JSONDecodeError as e:
         logger.error(f"Lỗi parse JSON file cấu hình cột {config_file_path}: {e}")
         raise HTTPException(status_code=500, detail=f"File cấu hình cột bị lỗi định dạng JSON.")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file cấu hình cột cho client {client_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi server khi đọc file cấu hình: {e}")