import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Any

from sklearn.ensemble import IsolationForest

# --- Project imports ---
from app.core.config import (
    OUTLIER_CONTAMINATION, # Tham số từ config
    OUTLIER_DETECTOR_1_FILENAME,
    OUTLIER_DETECTOR_2_FILENAME
)
from app.ml.utils import get_client_models_path, save_joblib, load_joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_outlier_detector(
    client_id: str,
    data: np.ndarray, # Dữ liệu đầu vào đã được transform (output của preprocessor)
    detector_filename: str
) -> Optional[IsolationForest]:
    """
    Huấn luyện mô hình Isolation Forest để phát hiện outlier.

    Args:
        client_id: ID của khách hàng.
        data: Dữ liệu features đã được tiền xử lý (numpy array hoặc sparse matrix).
        detector_filename: Tên file để lưu mô hình outlier detector.

    Returns:
        Mô hình IsolationForest đã huấn luyện hoặc None nếu lỗi.
    """
    if data is None or data.shape[0] == 0:
        logger.warning(f"Không có dữ liệu để huấn luyện outlier detector {detector_filename}.")
        return None

    # Kiểm tra xem data có phải là sparse matrix không và chuyển đổi nếu cần
    # Isolation Forest thường hoạt động tốt hơn với dense data.
    if hasattr(data, "toarray"):
        logger.info("Chuyển đổi sparse matrix sang dense array cho Isolation Forest.")
        try:
            # Cẩn thận với bộ nhớ nếu ma trận quá lớn
            data = data.toarray()
        except MemoryError:
            logger.error("Không đủ bộ nhớ để chuyển đổi sparse matrix sang dense. Huấn luyện outlier detector có thể không chính xác hoặc thất bại.")
            # Có thể thử dùng Incremental PCA trước hoặc giảm số features
            return None
        except Exception as e:
             logger.error(f"Lỗi khi chuyển đổi sparse matrix: {e}", exc_info=True)
             return None


    # Xử lý trường hợp chỉ có 1 mẫu dữ liệu
    if data.shape[0] <= 1:
        logger.warning(f"Chỉ có {data.shape[0]} mẫu dữ liệu, không thể huấn luyện Isolation Forest hiệu quả. Bỏ qua.")
        return None


    logger.info(f"Bắt đầu huấn luyện Isolation Forest detector: {detector_filename} với {data.shape[0]} mẫu và {data.shape[1]} features.")
    models_path = get_client_models_path(client_id)
    detector_path = models_path / detector_filename

    # Khởi tạo IsolationForest
    # contamination='auto' là cách tiếp cận tốt, nhưng có thể cần điều chỉnh
    # dựa trên dữ liệu thực tế hoặc đặt một giá trị cụ thể (ví dụ: 0.01 cho 1%)
    # random_state để đảm bảo kết quả tái lập được
    outlier_detector = IsolationForest(
        contamination=OUTLIER_CONTAMINATION,
        n_estimators=100, # Số lượng cây (có thể điều chỉnh)
        random_state=42,
        n_jobs=-1 # Sử dụng tất cả các CPU cores
    )

    try:
        # Huấn luyện mô hình
        outlier_detector.fit(data)
        logger.info("Isolation Forest đã huấn luyện xong.")

        # Lưu mô hình
        save_joblib(outlier_detector, detector_path)
        logger.info(f"Outlier detector đã được lưu vào: {detector_path}")
        return outlier_detector

    except ValueError as ve:
         # Bắt lỗi thường gặp khi contamination không hợp lệ hoặc dữ liệu có vấn đề
         logger.error(f"Lỗi ValueError khi huấn luyện Isolation Forest ({detector_filename}): {ve}. Có thể do 'contamination' hoặc dữ liệu đầu vào.")
         return None
    except Exception as e:
        logger.error(f"Lỗi không xác định khi huấn luyện Isolation Forest ({detector_filename}): {e}", exc_info=True)
        return None


def load_outlier_detector(client_id: str, detector_filename: str) -> Optional[IsolationForest]:
    """Tải mô hình outlier detector đã lưu."""
    models_path = get_client_models_path(client_id)
    detector_path = models_path / detector_filename
    logger.info(f"Đang tải outlier detector: {detector_path}")
    return load_joblib(detector_path)


def check_outlier(detector: Optional[IsolationForest], data: np.ndarray) -> List[bool]:
    """
    Kiểm tra xem các điểm dữ liệu mới có phải là outlier hay không.

    Args:
        detector: Mô hình IsolationForest đã huấn luyện.
        data: Dữ liệu features mới cần kiểm tra (đã được transform).

    Returns:
        List các giá trị boolean, True nếu là outlier, False nếu là inlier.
        Trả về list [False] * n nếu detector không tồn tại hoặc có lỗi.
    """
    if detector is None:
        logger.warning("Outlier detector chưa được huấn luyện hoặc không thể tải. Mặc định coi tất cả không phải outlier.")
        if data is None: return []
        return [False] * data.shape[0]

    if data is None or data.shape[0] == 0:
        logger.info("Không có dữ liệu để kiểm tra outlier.")
        return []

    # Chuyển đổi sang dense nếu cần (nhất quán với lúc train)
    if hasattr(data, "toarray"):
        logger.debug("Chuyển đổi sparse matrix sang dense array để dự đoán outlier.")
        try:
            data = data.toarray()
        except MemoryError:
            logger.error("Không đủ bộ nhớ để chuyển đổi sparse matrix sang dense cho dự đoán outlier. Mặc định không phải outlier.")
            return [False] * data.shape[0]
        except Exception as e:
             logger.error(f"Lỗi khi chuyển đổi sparse matrix cho dự đoán outlier: {e}. Mặc định không phải outlier.", exc_info=True)
             return [False] * data.shape[0]

    try:
        # predict trả về 1 cho inliers, -1 cho outliers
        predictions = detector.predict(data)
        # Chuyển đổi sang boolean: True nếu là outlier (-1), False nếu là inlier (1)
        # Chuyển đổi sang boolean chuẩn của Python

        is_outlier = [bool(pred == -1) for pred in predictions]
        num_outliers = sum(is_outlier)
        if num_outliers > 0:
             logger.info(f"Phát hiện {num_outliers}/{len(is_outlier)} điểm dữ liệu là outlier.")
        else:
             logger.debug("Không phát hiện outlier nào trong batch dữ liệu này.")
        return is_outlier
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán outlier: {e}", exc_info=True)
        # Trả về False cho tất cả nếu có lỗi
        return [False] * data.shape[0]