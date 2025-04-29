# app/ml/utils.py

import joblib
import logging
from pathlib import Path
from typing import Any, Optional
import app.core.config as config # Import config module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_client_data_path(client_id: str) -> Path:
    """Lấy đường dẫn đến thư mục dữ liệu của một client."""
    path = config.BASE_DATA_PATH / client_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_client_models_path(client_id: str) -> Path:
    """Lấy đường dẫn đến thư mục mô hình của một client."""
    path = config.BASE_MODELS_PATH / client_id
    path.mkdir(parents=True, exist_ok=True)
    # Tạo thư mục con cho label encoders nếu chưa có
    (path / config.LABEL_ENCODERS_DIR).mkdir(parents=True, exist_ok=True)
    return path

def get_client_label_encoder_path(client_id: str) -> Path:
    """Lấy đường dẫn đến thư mục lưu label encoder của client."""
    return get_client_models_path(client_id) / config.LABEL_ENCODERS_DIR

def save_joblib(data: Any, file_path: Path):
    """Lưu đối tượng Python sử dụng joblib."""
    try:
        # Đảm bảo thư mục cha tồn tại
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, file_path, compress=3) # Thêm nén để giảm kích thước
        logger.info(f"Đối tượng đã được lưu vào: {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu đối tượng vào {file_path}: {e}", exc_info=True)
        raise

def load_joblib(file_path: Path) -> Optional[Any]:
    """Tải đối tượng Python từ file joblib."""
    if not file_path.exists():
        logger.warning(f"File không tồn tại: {file_path}")
        return None
    try:
        data = joblib.load(file_path)
        logger.info(f"Đối tượng đã được tải từ: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Lỗi khi tải đối tượng từ {file_path}: {e}", exc_info=True)
        return None