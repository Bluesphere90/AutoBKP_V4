# app/ml/models.py

import pandas as pd
import numpy as np
import logging
import json
import time
import os # Đảm bảo import os
from datetime import datetime, timezone # Import timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable

# Import các lớp model cần hỗ trợ
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
# try: import lightgbm as lgb
# except ImportError: lgb = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone # Import clone
import scipy.sparse # Để kiểm tra sparse matrix

# --- Project imports ---
import app.core.config as config
from app.ml.utils import (
    get_client_models_path, get_client_label_encoder_path,
    save_joblib, load_joblib,
)
from app.ml.data_handler import load_all_client_data
# Import Pipeline Builder động
from app.ml.pipeline import build_dynamic_preprocessor
from app.ml.outlier_detector import (
    train_outlier_detector, load_outlier_detector, check_outlier
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function for JSON Serialization ---
def make_json_serializable(obj):
    """Chuyển đổi các kiểu dữ liệu không serializable sang dạng JSON chấp nhận."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None # Hoặc str(obj)
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Xử lý mảng numpy, đảm bảo các phần tử bên trong cũng được chuyển đổi
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (datetime, Path)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, bool):  # Xử lý bool chuẩn
        return obj  # Trả về chính nó (True/False)
    elif isinstance(obj, np.bool_): # Xử lý numpy bool
        return bool(obj)
    # Cố gắng chuyển đổi các kiểu numpy scalar khác
    try:
        if hasattr(obj, 'item'): # Check for numpy scalar types
             return obj.item()
    except Exception:
        pass # Bỏ qua nếu không phải numpy scalar

    # Nếu không thể chuyển đổi, trả về string hoặc None/error tùy logic
    # logger.warning(f"Could not serialize object of type {type(obj)}, converting to string.")
    # Kiểm tra các kiểu không thể serialize khác nếu cần
    if isinstance(obj, (type, Callable)): # Ví dụ: không serialize class hoặc function
        return str(obj)
    # Fallback cuối cùng
    try:
        # Thử json dump trực tiếp xem có lỗi không, nếu không thì trả về obj
        # Điều này nguy hiểm nếu obj là kiểu phức tạp không mong muốn
        # json.dumps(obj) # Chỉ để kiểm tra, không dùng kết quả
        # Tạm thời trả về string cho các trường hợp không xác định
        # Thử kiểm tra xem có thể dump không, nếu có thì giữ nguyên
        # Tránh chuyển đổi không cần thiết
        json.dumps(obj)
        return obj
    except TypeError:
        logger.warning(f"Could not serialize object of type {type(obj)}, returning its string representation.")
        return str(obj)

# --- Helper Function for Label Encoding ---
def _fit_or_load_label_encoder(client_id: str, target_series: pd.Series, encoder_filename: str) -> Optional[LabelEncoder]:
    """Fit LabelEncoder trên dữ liệu mới hoặc load cái đã có và cập nhật nếu cần."""
    encoder_path = get_client_label_encoder_path(client_id) / encoder_filename
    if target_series.empty:
        logger.warning(f"Không có dữ liệu target ({target_series.name}) để fit LabelEncoder.")
        if encoder_path.exists():
            try: encoder_path.unlink(); logger.info(f"Đã xóa LabelEncoder cũ: {encoder_path}")
            except OSError as e: logger.error(f"Lỗi khi xóa LabelEncoder cũ {encoder_path}: {e}")
        return None
    logger.info(f"Fit lại LabelEncoder cho {target_series.name}...")
    try:
        label_encoder = LabelEncoder()
        # Loại bỏ NaN trước khi fit và đảm bảo là string
        unique_labels = target_series.astype(str).dropna().unique()
        if len(unique_labels) == 0:
             logger.warning(f"Không có giá trị hợp lệ nào trong target series '{target_series.name}' để fit encoder.")
             if encoder_path.exists(): encoder_path.unlink() # Xóa file cũ nếu không fit được
             return None
        label_encoder.fit(unique_labels)
        save_joblib(label_encoder, encoder_path)
        logger.info(f"Đã fit và lưu LabelEncoder vào: {encoder_path}")
        return label_encoder
    except Exception as e:
        logger.error(f"Lỗi khi fit/lưu LabelEncoder cho {target_series.name}: {e}", exc_info=True)
        if encoder_path.exists():
            try: encoder_path.unlink()
            except OSError: pass
        return None

def _load_label_encoder(client_id: str, encoder_filename: str) -> Optional[LabelEncoder]:
    """Tải LabelEncoder đã lưu."""
    encoder_path = get_client_label_encoder_path(client_id) / encoder_filename
    return load_joblib(encoder_path)


# --- Function to get latest metadata file ---
def _find_latest_metadata_file(models_path: Path) -> Optional[Path]:
    """Tìm file metadata mới nhất trong thư mục models."""
    try:
        metadata_files = sorted(
            models_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"),
            key=os.path.getmtime, # Sắp xếp theo thời gian sửa đổi
            reverse=True
        )
        if metadata_files:
            return metadata_files[0]
    except Exception as e:
         logger.error(f"Lỗi khi tìm file metadata: {e}")
    return None

# --- Function to load model type from metadata ---
def _load_model_type_from_metadata(metadata_file: Path) -> Optional[str]:
    """Đọc loại model đã lưu từ file metadata."""
    if not metadata_file or not metadata_file.exists():
        return None
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Ưu tiên key 'selected_model_type' được lưu ở cấp cao nhất
        model_type = metadata.get("selected_model_type")
        # Fallback: thử lấy từ hachtoan_model_info nếu key trên không có
        if not model_type and "hachtoan_model_info" in metadata:
            model_type = metadata["hachtoan_model_info"].get("model_class")

        if model_type and isinstance(model_type, str):
             # Kiểm tra xem model_type có nằm trong danh sách hỗ trợ không
             if model_type in [e.value for e in config.SupportedModels]:
                 logger.info(f"Đã đọc model type '{model_type}' từ metadata: {metadata_file.name}")
                 return model_type
             else:
                  logger.warning(f"Model type '{model_type}' đọc từ metadata không được hỗ trợ.")
                  return None
        else:
             logger.warning(f"Không tìm thấy hoặc định dạng key model type không đúng trong metadata: {metadata_file.name}")
             return None
    except Exception as e:
        logger.error(f"Lỗi khi đọc model type từ metadata {metadata_file.name}: {e}")
        return None

# --- Function to instantiate model ---
def _instantiate_model(model_type_str: str) -> Optional[Any]:
    """Khởi tạo đối tượng model dựa trên tên và tham số mặc định từ config."""
    logger.info(f"Khởi tạo model loại: {model_type_str}")
    # Lấy tham số mặc định từ config
    params = config.DEFAULT_MODEL_PARAMS.get(model_type_str, {})
    logger.debug(f"Sử dụng tham số mặc định: {params}")

    try:
        if model_type_str == config.SupportedModels.RANDOM_FOREST.value:
            # Đảm bảo params hợp lệ cho RandomForest
            rf_params = {k:v for k,v in params.items() if k in RandomForestClassifier().get_params()}
            return RandomForestClassifier(**rf_params)
        elif model_type_str == config.SupportedModels.LOGISTIC_REGRESSION.value:
            lr_params = {k:v for k,v in params.items() if k in LogisticRegression().get_params()}
            return LogisticRegression(**lr_params)
        elif model_type_str == config.SupportedModels.MULTINOMIAL_NB.value:
            nb_params = {k:v for k,v in params.items() if k in MultinomialNB().get_params()}
            return MultinomialNB(**nb_params)
        elif model_type_str == config.SupportedModels.LINEAR_SVC.value:
            svc_params = {k:v for k,v in params.items() if k in LinearSVC().get_params()}
            # Xử lý tham số 'dual' đặc biệt cho LinearSVC dựa trên n_samples và n_features
            # Tuy nhiên, trong hàm khởi tạo đơn giản này, chúng ta có thể bỏ qua hoặc đặt mặc định an toàn
            svc_params.setdefault('dual', 'auto') # Đặt mặc định an toàn
            return LinearSVC(**svc_params)
        # elif model_type_str == config.SupportedModels.LIGHTGBM.value:
        #     if lgb: return lgb.LGBMClassifier(**params) else: logger.error("LightGBM not installed."); return None
        else:
            logger.error(f"Loại model không xác định hoặc không được hỗ trợ: {model_type_str}")
            return None
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo model {model_type_str} với params {params}: {e}", exc_info=True)
        return None


# --- Core Training Logic (KHÔNG fit preprocessor) ---
def _train_and_evaluate_model(
    client_id: str,
    X: pd.DataFrame, # Dữ liệu input đã chuẩn bị
    y: pd.Series,    # Dữ liệu target đã chuẩn bị
    target_column_name: str, # Tên cột target để log
    preprocessor: ColumnTransformer, # Preprocessor ĐÃ FIT được truyền vào
    model_object: Any, # Model instance đã khởi tạo
    model_save_filename: str, # Tên file để lưu model này
    encoder: LabelEncoder, # Encoder ĐÃ FIT được truyền vào
    outlier_detector_filename: str, # Tên file lưu outlier detector
    validation_size: float = config.VALIDATION_SET_SIZE
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]: # Chỉ trả về model đã fit và metrics
    """
    Huấn luyện model object, outlier detector, đánh giá, và trả về metrics.
    Sử dụng preprocessor và encoder đã được fit từ bên ngoài.
    """
    models_path = get_client_models_path(client_id)
    model_path = models_path / model_save_filename
    outlier_path = models_path / outlier_detector_filename

    logger.info(f"Bắt đầu huấn luyện model '{type(model_object).__name__}' cho target: '{target_column_name}' với {len(X)} bản ghi.")

    # --- Transform dữ liệu bằng preprocessor đã fit ---
    logger.info("Transforming dữ liệu với preprocessor đã fit...")
    X_transformed = None
    try:
        # Cần đảm bảo X có các cột mà preprocessor mong đợi
        expected_cols = getattr(preprocessor, 'feature_names_in_', None)
        if expected_cols is not None:
            missing_in_X = set(expected_cols) - set(X.columns)
            if missing_in_X:
                 logger.warning(f"Input X thiếu các cột mà preprocessor mong đợi: {missing_in_X}. Thêm cột với giá trị rỗng.")
                 for col in missing_in_X: X[col] = "" # Thêm cột thiếu
            # Sắp xếp lại và chỉ giữ các cột mong đợi
            X_aligned = X[list(expected_cols)]
        else:
             logger.warning("Không thể lấy feature_names_in_ từ preprocessor. Sử dụng X gốc.")
             X_aligned = X

        X_transformed = preprocessor.transform(X_aligned)
        logger.info(f"Dữ liệu đã transform. Output type: {type(X_transformed)}, Shape: {X_transformed.shape if hasattr(X_transformed, 'shape') else 'N/A'}")
    except Exception as e:
        logger.error(f"Lỗi khi transform dữ liệu cho target {target_column_name}: {e}", exc_info=True)
        return None, {"error": f"Transform failed: {e}"}

    # --- Encode target ---
    y_encoded = None
    try: y_encoded = encoder.transform(y.astype(str))
    except ValueError as e:
         # Lỗi này thường xảy ra nếu y chứa nhãn chưa từng thấy trong lúc fit encoder
         logger.error(f"Lỗi khi transform target '{target_column_name}' bằng encoder đã fit: {e}.")
         return None, {"error": f"Target encoding failed: {e}"}
    num_classes = len(encoder.classes_)
    if num_classes == 0:
         logger.error(f"Encoder cho target '{target_column_name}' không có lớp nào.")
         return None, {"error": "Encoder has no classes."}

    # --- Huấn luyện Outlier Detector ---
    if X_transformed is not None and X_transformed.shape[0] > 1:
        logger.info(f"Huấn luyện Outlier Detector ({outlier_detector_filename})...")
        _ = train_outlier_detector(client_id=client_id, data=X_transformed, detector_filename=outlier_detector_filename)
    else:
        logger.warning("Bỏ qua huấn luyện Outlier Detector do dữ liệu transform không hợp lệ.")
        if outlier_path.exists(): outlier_path.unlink()

    # --- Đánh giá mô hình ---
    metrics = None
    final_model = model_object
    if num_classes >= 2 and validation_size > 0 and validation_size < 1 and X_transformed is not None and X_transformed.shape[0] > 1:
        logger.info(f"Thực hiện đánh giá mô hình {type(final_model).__name__}...")
        try:
            unique_classes_y, counts_y = np.unique(y_encoded, return_counts=True)
            min_samples_per_class = counts_y.min() if len(counts_y) > 0 else 0
            can_stratify = min_samples_per_class >= 2
            if not can_stratify: logger.warning(f"Không đủ mẫu/lớp ({min_samples_per_class}) để stratify.")

            X_train_eval, X_val, y_train_eval, y_val = train_test_split(
                X_transformed, y_encoded, test_size=validation_size, random_state=42,
                stratify=y_encoded if can_stratify else None
            )
            if X_train_eval.shape[0] == 0 or X_val.shape[0] == 0:
                 logger.warning("Tập train/validation rỗng. Bỏ qua đánh giá.")
                 metrics = {"warning": "Skipped evaluation due to empty split."}
            else:
                eval_model = clone(final_model); eval_model.fit(X_train_eval, y_train_eval)
                y_pred_val = eval_model.predict(X_val)
                present_labels_indices = np.unique(np.concatenate((y_val, y_pred_val)))
                present_labels_names = []
                if len(encoder.classes_) > 0:
                     try: present_labels_names = encoder.inverse_transform(present_labels_indices)
                     except ValueError: logger.warning("Lỗi inverse_transform labels."); present_labels_names = [str(i) for i in present_labels_indices]
                report = classification_report(
                    y_val, y_pred_val,
                    labels=present_labels_indices,
                    # Chỉ truyền target_names nếu list không rỗng
                    target_names=list(present_labels_names) if len(present_labels_names) > 0 else None,
                    output_dict=True, zero_division=0
                )
                if len(present_labels_indices) < num_classes:
                     missing_labels_indices = np.setdiff1d(np.arange(num_classes), present_labels_indices)
                     if len(encoder.classes_) > 0:
                          try: missing_labels_names = encoder.inverse_transform(missing_labels_indices); report['missing_in_validation'] = list(missing_labels_names)
                          except ValueError: report['missing_in_validation'] = [str(i) for i in missing_labels_indices]
                     else: report['missing_in_validation'] = []
                logger.info(f"Kết quả đánh giá (Validation Set) cho {target_column_name}:\n{json.dumps(make_json_serializable(report), indent=2)}")
                metrics = report
        except Exception as e: logger.error(f"Lỗi đánh giá {target_column_name}: {e}", exc_info=True); metrics = {"error": f"Evaluation failed: {e}"}

    # --- Huấn luyện mô hình cuối cùng ---
    logger.info(f"Huấn luyện mô hình cuối cùng {type(final_model).__name__} cho {target_column_name}...")
    fitted_final_model = None
    if num_classes < 2 or X_transformed is None or X_transformed.shape[0] == 0:
         logger.warning(f"Không đủ lớp hoặc dữ liệu không hợp lệ. Không huấn luyện model cuối.")
         if model_path.exists(): model_path.unlink()
    else:
        try:
            final_model.fit(X_transformed, y_encoded)
            logger.info(f"Mô hình cuối cùng {type(final_model).__name__} đã huấn luyện xong.")
            save_joblib(final_model, model_path)
            logger.info(f"Mô hình cuối cùng đã được lưu vào: {model_path}")
            fitted_final_model = final_model
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình cuối cùng: {e}", exc_info=True)
            if model_path.exists(): model_path.unlink()

    # Trả về model đã fit và metrics
    return fitted_final_model, metrics


# --- Main Training Orchestrator ---
def train_client_models(client_id: str, initial_model_type_str: Optional[str] = None) -> bool:
    """
    Huấn luyện/Tái huấn luyện các mô hình và lưu metadata.
    Fit preprocessor chỉ lần đầu.
    """
    start_time = time.time()
    training_timestamp_utc = datetime.now(timezone.utc)
    logger.info(f"===== Bắt đầu quy trình huấn luyện cho client: {client_id} lúc {training_timestamp_utc} UTC =====")
    models_path = get_client_models_path(client_id)
    metadata = { # Khởi tạo metadata
        "client_id": client_id, "training_timestamp_utc": training_timestamp_utc.isoformat(),
        "status": "STARTED", "training_type": None, "selected_model_type": None,
        "column_config_file": None, "data_info": {},
        "hachtoan_model_info": {}, "mahanghoa_model_info": {}, "mahanghoa_direct_model_info": {},
        "training_duration_seconds": None, "error_message": None
    }
    meta_filename_placeholder = f"{config.METADATA_FILENAME_PREFIX}{training_timestamp_utc.strftime('%Y%m%d_%H%M%S')}.json"
    def save_final_metadata(status="FAILED"): # Hàm helper
        metadata["status"] = status
        metadata["training_duration_seconds"] = round(time.time() - start_time, 2)
        try:
            serializable_metadata = make_json_serializable(metadata)
            with open(models_path / meta_filename_placeholder, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"Đã lưu metadata ({status}) vào {meta_filename_placeholder}")
        except Exception as json_e: logger.error(f"Lỗi khi lưu metadata cuối: {json_e}")

    # --- Load Column Config ---
    column_config_path = models_path / "column_config.json"
    column_metadata = None
    if not column_config_path.exists(): error_msg = f"Thiếu file column_config.json"; logger.error(error_msg); save_final_metadata(error_msg); return False
    try:
        with open(column_config_path, 'r', encoding='utf-8') as f: column_metadata = json.load(f)
        metadata["column_config_file"] = column_config_path.name; logger.info(f"Đã tải config cột.")
        if "columns" not in column_metadata: raise ValueError("Thiếu key 'columns'.")
    except Exception as e: error_msg = f"Lỗi đọc/parse column_config.json: {e}"; logger.error(error_msg, exc_info=True); save_final_metadata(error_msg); return False

    # --- Xác định Training Type và Model Type ---
    latest_metadata_file = _find_latest_metadata_file(models_path)
    model_type_to_train = None; is_initial_training = False
    if initial_model_type_str:
        metadata["training_type"] = "initial"; is_initial_training = True
        if initial_model_type_str in [e.value for e in config.SupportedModels]:
             model_type_to_train = initial_model_type_str; metadata["selected_model_type"] = model_type_to_train
             logger.info(f"Huấn luyện ban đầu, model: {model_type_to_train}")
        else: error_msg = f"Loại model '{initial_model_type_str}' không hợp lệ."; logger.error(error_msg); save_final_metadata(error_msg); return False
    elif latest_metadata_file:
        metadata["training_type"] = "incremental"
        saved_model_type = _load_model_type_from_metadata(latest_metadata_file)
        if saved_model_type: model_type_to_train = saved_model_type
        else: model_type_to_train = config.DEFAULT_MODEL_TYPE.value; logger.warning("Dùng model mặc định.")
        metadata["selected_model_type"] = model_type_to_train
        logger.info(f"Huấn luyện tăng cường, model: {model_type_to_train}")
    else: error_msg = "Không thể xác định loại model."; logger.error(error_msg); save_final_metadata(error_msg); return False

    # --- Xác định data files ---
    client_data_path = config.BASE_DATA_PATH / client_id
    initial_data_file = client_data_path / config.TRAINING_DATA_FILENAME
    incremental_files = sorted(client_data_path.glob(f"{config.INCREMENTAL_DATA_PREFIX}*.csv"))
    data_files_used = []
    if initial_data_file.exists():
        data_files_used.append(initial_data_file.name)
    # ---------------------------

    if incremental_files:
        data_files_used.extend([f.name for f in incremental_files])
    if not data_files_used:  # Nếu không có file nào cả
        metadata["training_type"] = "unknown"
    # Cập nhật metadata sau khi đã xác định xong các file
    metadata["data_info"]["files_used"] = data_files_used

    if incremental_files: data_files_used.extend([f.name for f in incremental_files])
    if not data_files_used: metadata["training_type"] = "unknown"
    metadata["data_info"]["files_used"] = data_files_used

    # 1. Load tất cả dữ liệu
    df = load_all_client_data(client_id)
    if df is None or df.empty:
        error_msg = f"Không có dữ liệu huấn luyện hợp lệ."; logger.error(error_msg); save_final_metadata(error_msg); return False
    metadata["data_info"]["total_samples_loaded"] = len(df)
    metadata["data_info"]["columns_present"] = df.columns.tolist()

    # --- Khởi tạo model instances ---
    model_ht_instance = _instantiate_model(model_type_to_train)
    model_mh_instance = _instantiate_model(model_type_to_train)
    model_mh_direct_instance = _instantiate_model(model_type_to_train)
    if model_ht_instance is None or model_mh_instance is None or model_mh_direct_instance is None:
        error_msg = f"Không thể khởi tạo model '{model_type_to_train}'."; logger.error(error_msg); save_final_metadata(error_msg); return False

    # --- Đặt tên file ---
    preprocessor_ht_save_file = config.PREPROCESSOR_HACHTOAN_FILENAME
    preprocessor_mh_save_file = config.PREPROCESSOR_MAHANGHOA_FILENAME
    hachtoan_model_save_file = config.HACHTOAN_MODEL_FILENAME
    mahanghoa_model_save_file = config.MAHANGHOA_MODEL_FILENAME
    mahanghoa_direct_model_save_file = config.MAHANGHOA_DIRECT_MODEL_FILENAME
    hachtoan_encoder_file = config.HACHTOAN_ENCODER_FILENAME
    mahanghoa_encoder_file = config.MAHANGHOA_ENCODER_FILENAME
    outlier_detector_1_file = config.OUTLIER_DETECTOR_1_FILENAME
    outlier_detector_2_file = config.OUTLIER_DETECTOR_2_FILENAME

    # --- Xử lý Preprocessors ---
    preprocessor_ht = None; preprocessor_mh = None
    preprocessor_ht_path = models_path / preprocessor_ht_save_file
    preprocessor_mh_path = models_path / preprocessor_mh_save_file

    if is_initial_training or not preprocessor_ht_path.exists() or not preprocessor_mh_path.exists():
        logger.info("--- Fit và Lưu Preprocessors (Lần đầu hoặc file bị thiếu) ---")
        # Xóa file cũ nếu có (để fit lại từ đầu)
        if preprocessor_ht_path.exists(): preprocessor_ht_path.unlink()
        if preprocessor_mh_path.exists(): preprocessor_mh_path.unlink()

        cols_ht_meta = {k: v for k, v in column_metadata.get("columns", {}).items() if k not in [config.TARGET_MAHANGHOA, config.TARGET_HACHTOAN]}
        metadata_ht_prep = {"columns": cols_ht_meta, **{k:v for k,v in column_metadata.items() if k != 'columns'}}
        input_cols_ht = [col for col in metadata_ht_prep["columns"] if col in df.columns]

        cols_mh_meta = {k: v for k, v in column_metadata.get("columns", {}).items() if k != config.TARGET_MAHANGHOA}
        if config.TARGET_HACHTOAN not in cols_mh_meta and config.TARGET_HACHTOAN in df.columns:
             cols_mh_meta[config.TARGET_HACHTOAN] = {"type": "categorical", "strategy": "onehot"}
        metadata_mh_prep = {"columns": cols_mh_meta, **{k:v for k,v in column_metadata.items() if k != 'columns'}}
        input_cols_mh = [col for col in metadata_mh_prep["columns"] if col in df.columns]

        # Fit HT Preprocessor
        if input_cols_ht:
            try:
                X_ht_prep = df[input_cols_ht].copy().dropna() # Dropna trên input trước khi fit prep? Có thể không cần.
                if not X_ht_prep.empty:
                    preprocessor_ht = build_dynamic_preprocessor(metadata_ht_prep)
                    logger.info(f"Fitting preprocessor HachToan trên {len(X_ht_prep)} mẫu...")
                    preprocessor_ht.fit(X_ht_prep) # Fit chỉ với X
                    save_joblib(preprocessor_ht, preprocessor_ht_path)
                    logger.info(f"Đã lưu preprocessor HachToan: {preprocessor_ht_path.name}")
                else: logger.warning("Không có dữ liệu hợp lệ để fit preprocessor HachToan.")
            except Exception as e: error_msg = f"Lỗi fit/lưu prep HT: {e}"; logger.error(error_msg, exc_info=True); save_final_metadata(error_msg); return False
        else: error_msg = "Không có cột input cho prep HT."; logger.error(error_msg); save_final_metadata(error_msg); return False

        # Fit MH Preprocessor
        df_mh_for_prep = pd.DataFrame()
        if input_cols_mh and config.TARGET_HACHTOAN in df.columns: # Cần HachToan để lọc
            df_mh_for_prep = df[df[config.TARGET_HACHTOAN].astype(str).str.startswith(config.HACHTOAN_PREFIX_FOR_MAHANGHOA)].copy()
            # Chỉ giữ lại các cột input cần thiết cho preprocessor MH
            missing_mh_input_cols = [col for col in input_cols_mh if col not in df_mh_for_prep.columns]
            if missing_mh_input_cols:
                 logger.warning(f"Dữ liệu lọc cho MH prep thiếu cột: {missing_mh_input_cols}. Bỏ qua các cột này.")
                 input_cols_mh_present = [col for col in input_cols_mh if col in df_mh_for_prep.columns]
            else:
                 input_cols_mh_present = input_cols_mh

            if input_cols_mh_present and not df_mh_for_prep.empty:
                df_mh_for_prep = df_mh_for_prep[input_cols_mh_present].dropna() # Dropna trên input

        if not df_mh_for_prep.empty:
            try:
                preprocessor_mh = build_dynamic_preprocessor(metadata_mh_prep)
                logger.info(f"Fitting preprocessor MaHangHoa trên {len(df_mh_for_prep)} mẫu...")
                preprocessor_mh.fit(df_mh_for_prep)
                save_joblib(preprocessor_mh, preprocessor_mh_path)
                logger.info(f"Đã lưu preprocessor MaHangHoa: {preprocessor_mh_path.name}")
            except Exception as e:
                logger.error(f"Lỗi khi fit/lưu preprocessor MaHangHoa: {e}", exc_info=True)
                metadata["mahanghoa_model_info"]["preprocessor_error"] = str(e)
                preprocessor_mh = None
                if preprocessor_mh_path.exists(): preprocessor_mh_path.unlink()
        else:
            logger.warning("Không có dữ liệu phù hợp để fit preprocessor MaHangHoa.")
            if preprocessor_mh_path.exists(): preprocessor_mh_path.unlink()

    else: # Huấn luyện tăng cường - Load preprocessors
        logger.info("--- Load Preprocessors (Huấn luyện tăng cường) ---")
        preprocessor_ht = load_joblib(preprocessor_ht_path)
        preprocessor_mh = load_joblib(preprocessor_mh_path)
        if preprocessor_ht is None: error_msg = f"Không thể load prep HT."; logger.error(error_msg); save_final_metadata(error_msg); return False
        if preprocessor_mh is None: logger.warning(f"Không thể load prep MH.")

    # --- Huấn luyện các model ---
    training_successful = True # Cờ theo dõi thành công

    # 2. Huấn luyện HachToan Model
    logger.info(f"--- Huấn luyện mô hình {model_type_to_train} cho HachToan ---")
    if preprocessor_ht:
        df_ht_train = df.dropna(subset=[config.TARGET_HACHTOAN]).copy()
        input_cols_ht_train = [col for col in getattr(preprocessor_ht, 'feature_names_in_', df_ht_train.columns) if col in df_ht_train.columns and col != config.TARGET_HACHTOAN]
        if input_cols_ht_train and not df_ht_train.empty:
            X_ht_train = df_ht_train[input_cols_ht_train]; y_ht_train = df_ht_train[config.TARGET_HACHTOAN]
            encoder_ht = _fit_or_load_label_encoder(client_id, y_ht_train, hachtoan_encoder_file)
            if encoder_ht:
                model_ht, metrics_ht = _train_and_evaluate_model(
                    client_id, X_ht_train, y_ht_train, config.TARGET_HACHTOAN,
                    preprocessor_ht, model_ht_instance, hachtoan_model_save_file,
                    encoder_ht, outlier_detector_1_file)
                metadata["hachtoan_model_info"].update({ # Cập nhật metadata HT
                    "preprocessor_saved": True, "preprocessor_file": preprocessor_ht_save_file,
                    "encoder_saved": True, "model_saved": model_ht is not None,
                    "outlier_detector_saved": (models_path / outlier_detector_1_file).exists(),
                    "model_class": type(model_ht).__name__ if model_ht else None,
                    "model_params": model_ht.get_params() if model_ht else None,
                    "features_in": getattr(preprocessor_ht, 'feature_names_in_', None),
                    "num_classes": len(encoder_ht.classes_), "evaluation_metrics": metrics_ht
                })
            else: logger.error("HT Encoder failed."); training_successful = False
        else: logger.warning("Không có data huấn luyện HT."); metadata["hachtoan_model_info"]["message"] = "No valid data."
    else: logger.error("Prep HT không khả dụng."); training_successful = False; metadata["hachtoan_model_info"]["error"] = "Preprocessor not available."

    # 3. Huấn luyện MaHangHoa Model (dựa trên HachToan)
    logger.info(f"--- Huấn luyện mô hình {model_type_to_train} cho MaHangHoa (Dependent) ---")
    metadata["mahanghoa_model_info"]["attempted"] = False
    if preprocessor_mh and config.TARGET_MAHANGHOA in df.columns:
        df_mh_train = pd.DataFrame(); df_filtered_prefix = df[df[config.TARGET_HACHTOAN].astype(str).str.startswith(config.HACHTOAN_PREFIX_FOR_MAHANGHOA)].copy()
        if not df_filtered_prefix.empty: df_mh_train = df_filtered_prefix.dropna(subset=[config.TARGET_MAHANGHOA]).copy()
        if not df_mh_train.empty: df_mh_train = df_mh_train[df_mh_train[config.TARGET_MAHANGHOA].astype(str) != '']
        metadata["data_info"]["samples_for_mahanghoa_dependent"] = len(df_mh_train)
        if not df_mh_train.empty:
            metadata["mahanghoa_model_info"]["attempted"] = True
            input_cols_mh_train = [col for col in getattr(preprocessor_mh, 'feature_names_in_', df_mh_train.columns) if col in df_mh_train.columns and col != config.TARGET_MAHANGHOA]
            if input_cols_mh_train and config.TARGET_HACHTOAN in input_cols_mh_train:
                X_mh_train = df_mh_train[input_cols_mh_train]; y_mh_train = df_mh_train[config.TARGET_MAHANGHOA]
                encoder_mh = _fit_or_load_label_encoder(client_id, y_mh_train, mahanghoa_encoder_file)
                if encoder_mh:
                    model_mh, metrics_mh = _train_and_evaluate_model(
                        client_id, X_mh_train, y_mh_train, f"{config.TARGET_MAHANGHOA} (Dependent)",
                        preprocessor_mh, model_mh_instance, mahanghoa_model_save_file,
                        encoder_mh, outlier_detector_2_file)
                    metadata["mahanghoa_model_info"].update({ # Cập nhật metadata MH
                        "preprocessor_saved": True, "preprocessor_file": preprocessor_mh_save_file,
                        "encoder_saved": True, "model_saved": model_mh is not None,
                        "outlier_detector_saved": (models_path / outlier_detector_2_file).exists(),
                        "model_class": type(model_mh).__name__ if model_mh else None,
                        "model_params": model_mh.get_params() if model_mh else None,
                        "features_in": getattr(preprocessor_mh, 'feature_names_in_', None),
                        "num_classes": len(encoder_mh.classes_), "evaluation_metrics": metrics_mh
                    })
                else: logger.error("MH Encoder failed."); metadata["mahanghoa_model_info"]["error"] = "Encoder failed."
            else: logger.warning("Thiếu input/HachToan cho model MH."); metadata["mahanghoa_model_info"]["error"] = "Missing input columns."
        else:
            logger.warning(f"Không có data huấn luyện MH (Dependent).")
            metadata["mahanghoa_model_info"]["attempted"] = True; metadata["mahanghoa_model_info"]["message"] = "No valid data."
            if (models_path / mahanghoa_model_save_file).exists(): (models_path / mahanghoa_model_save_file).unlink()
            # ... (xóa encoder MH, outlier 2 nếu cần) ...
    else: logger.warning("Bỏ qua huấn luyện MH (Dependent) do thiếu prep MH hoặc cột target.")

    # 4. Huấn luyện MaHangHoa Direct Model
    logger.info(f"--- Huấn luyện mô hình {model_type_to_train} cho MaHangHoa (Direct) ---")
    metadata["mahanghoa_direct_model_info"]["attempted"] = False
    if preprocessor_ht and config.TARGET_MAHANGHOA in df.columns:
        df_mh_direct_train = df.dropna(subset=[config.TARGET_MAHANGHOA]).copy()
        if not df_mh_direct_train.empty: df_mh_direct_train = df_mh_direct_train[df_mh_direct_train[config.TARGET_MAHANGHOA].astype(str) != '']
        metadata["data_info"]["samples_for_mahanghoa_direct"] = len(df_mh_direct_train)
        if not df_mh_direct_train.empty:
            metadata["mahanghoa_direct_model_info"]["attempted"] = True
            input_cols_mh_direct_train = [col for col in getattr(preprocessor_ht, 'feature_names_in_', df_mh_direct_train.columns) if col in df_mh_direct_train.columns and col not in [config.TARGET_HACHTOAN, config.TARGET_MAHANGHOA]]
            if input_cols_mh_direct_train:
                X_mh_direct_train = df_mh_direct_train[input_cols_mh_direct_train]
                y_mh_direct_train = df_mh_direct_train[config.TARGET_MAHANGHOA]
                encoder_mh_direct = _fit_or_load_label_encoder(client_id, y_mh_direct_train, mahanghoa_encoder_file) # Dùng chung encoder MH
                if encoder_mh_direct:
                    model_mh_direct, metrics_mh_direct = _train_and_evaluate_model(
                        client_id, X_mh_direct_train, y_mh_direct_train, f"{config.TARGET_MAHANGHOA} (Direct)",
                        preprocessor_ht, model_mh_direct_instance, mahanghoa_direct_model_save_file,
                        encoder_mh_direct, outlier_detector_1_file # Dùng OD1
                    )
                    metadata["mahanghoa_direct_model_info"].update({ # Cập nhật metadata MH Direct
                        "preprocessor_used": preprocessor_ht_save_file, "encoder_saved": True,
                        "model_saved": model_mh_direct is not None, "outlier_detector_used": outlier_detector_1_file,
                        "model_class": type(model_mh_direct).__name__ if model_mh_direct else None,
                        "model_params": model_mh_direct.get_params() if model_mh_direct else None,
                        "features_in": getattr(preprocessor_ht, 'feature_names_in_', None),
                        "num_classes": len(encoder_mh_direct.classes_), "evaluation_metrics": metrics_mh_direct
                    })
                else: logger.error("MH Encoder (Direct) failed."); metadata["mahanghoa_direct_model_info"]["error"] = "Encoder failed."
            else: logger.warning("Thiếu input cho model MH Direct."); metadata["mahanghoa_direct_model_info"]["error"] = "Missing input columns."
        else:
            logger.warning(f"Không có data huấn luyện MH Direct.")
            metadata["mahanghoa_direct_model_info"]["attempted"] = True; metadata["mahanghoa_direct_model_info"]["message"] = "No valid data."
            if (models_path / mahanghoa_direct_model_save_file).exists(): (models_path / mahanghoa_direct_model_save_file).unlink()
            # Không xóa encoder/OD1
    else: logger.warning("Bỏ qua huấn luyện MH Direct do thiếu prep HT hoặc cột target.")


    # --- Hoàn tất và Lưu Metadata ---
    final_status = "COMPLETED" if training_successful else "COMPLETED_WITH_ERRORS"
    save_final_metadata(status=final_status)
    logger.info(f"===== Quy trình huấn luyện cho client {client_id} đã hoàn tất ({metadata['training_duration_seconds']:.2f}s). Status: {final_status} =====")
    # Thành công nếu preprocessor HT và encoder HT được tạo/load
    return preprocessor_ht is not None and (get_client_label_encoder_path(client_id) / hachtoan_encoder_file).exists()


# --- Prediction Logic (Sử dụng 2 preprocessors) ---
def _load_prediction_components(client_id: str) -> Dict[str, Optional[Any]]:
    """Tải tất cả các thành phần cần thiết cho các kịch bản dự đoán."""
    models_path = get_client_models_path(client_id)
    components = {
        "preprocessor_ht": load_joblib(models_path / config.PREPROCESSOR_HACHTOAN_FILENAME),
        "preprocessor_mh": load_joblib(models_path / config.PREPROCESSOR_MAHANGHOA_FILENAME),
        "model_ht": load_joblib(models_path / config.HACHTOAN_MODEL_FILENAME),
        "model_mh": load_joblib(models_path / config.MAHANGHOA_MODEL_FILENAME),
        "model_mh_direct": load_joblib(models_path / config.MAHANGHOA_DIRECT_MODEL_FILENAME),
        "encoder_ht": _load_label_encoder(client_id, config.HACHTOAN_ENCODER_FILENAME),
        "encoder_mh": _load_label_encoder(client_id, config.MAHANGHOA_ENCODER_FILENAME),
        "outlier_detector_1": load_outlier_detector(client_id, config.OUTLIER_DETECTOR_1_FILENAME),
        "outlier_detector_2": load_outlier_detector(client_id, config.OUTLIER_DETECTOR_2_FILENAME),
    }
    # Log nếu thiếu thành phần quan trọng
    if not components["preprocessor_ht"] or not components["encoder_ht"]: logger.error(f"Client {client_id}: Thiếu prep HT hoặc enc HT.")
    if not components["encoder_mh"]: logger.warning(f"Client {client_id}: Thiếu enc MH.")
    if not components["model_ht"]: logger.warning(f"Client {client_id}: Thiếu model HT.")
    if not components["preprocessor_mh"]: logger.warning(f"Client {client_id}: Thiếu prep MH (cần cho predict MH dependent).")
    if not components["model_mh"]: logger.warning(f"Client {client_id}: Thiếu model MH (dependent).")
    if not components["model_mh_direct"]: logger.warning(f"Client {client_id}: Thiếu model MH Direct.")
    return components

# --- Hàm dự đoán kết hợp (Model 1 -> Model 2) ---
def predict_combined(client_id: str, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dự đoán kết hợp HachToan -> MaHangHoa."""
    logger.info(f"Bắt đầu dự đoán kết hợp cho client {client_id}.")
    components = _load_prediction_components(client_id)
    preprocessor_ht = components["preprocessor_ht"]
    preprocessor_mh = components["preprocessor_mh"]
    encoder_ht = components["encoder_ht"]
    encoder_mh = components["encoder_mh"]
    model_ht = components["model_ht"]
    model_mh = components["model_mh"]
    od1 = components["outlier_detector_1"]
    od2 = components["outlier_detector_2"]

    n_items = len(input_data)
    results = [{} for _ in range(n_items)] # Khởi tạo list kết quả

    # --- Bước 1: Dự đoán HachToan ---
    y_pred_ht = [None]*n_items; prob_ht = [None]*n_items; flags_od1 = [False]*n_items; err_ht = [None]*n_items
    if not preprocessor_ht or not encoder_ht:
        for i in range(n_items): results[i].update({"error": "Thiếu thành phần HT."})
    else:
        try:
            expected_cols_ht = getattr(preprocessor_ht, 'feature_names_in_', None)
            if expected_cols_ht: input_data_aligned_ht = input_data.reindex(columns=expected_cols_ht, fill_value="")
            else: input_data_aligned_ht = input_data
            X_transformed_ht = preprocessor_ht.transform(input_data_aligned_ht)
            if od1: flags_od1 = check_outlier(od1, X_transformed_ht)
            if model_ht:
                pred_enc = model_ht.predict(X_transformed_ht); probs = model_ht.predict_proba(X_transformed_ht)
                y_pred_ht = encoder_ht.inverse_transform(pred_enc); prob_ht = np.max(probs, axis=1)
            elif len(encoder_ht.classes_) == 1: y_pred_ht = [encoder_ht.classes_[0]]*n_items; prob_ht = [1.0]*n_items
            else: err_ht = ["Lỗi: Model HT không tồn tại"]*n_items
        except Exception as e: err_ht = [f"Lỗi dự đoán HT: {e}"]*n_items

    # Gán kết quả bước 1
    for i in range(n_items):
        p_ht = prob_ht[i] if i < len(prob_ht) and prob_ht[i] is not None else None # Kiểm tra index và None
        if isinstance(p_ht, np.float64): p_ht = float(p_ht)
        results[i].update({
            config.TARGET_HACHTOAN: y_pred_ht[i] if i < len(y_pred_ht) else None,
            f"{config.TARGET_HACHTOAN}_prob": p_ht,
            config.TARGET_MAHANGHOA: None, f"{config.TARGET_MAHANGHOA}_prob": None,
            "is_outlier_input1": flags_od1[i] if i < len(flags_od1) else False,
            "is_outlier_input2": False, "error": err_ht[i] if i < len(err_ht) else None
        })

    # --- Bước 2: Dự đoán MaHangHoa (nếu cần và có thể) ---
    indices_to_predict_mh = [i for i, r in enumerate(results) if r[config.TARGET_HACHTOAN] and isinstance(r[config.TARGET_HACHTOAN], str) and r[config.TARGET_HACHTOAN].startswith(config.HACHTOAN_PREFIX_FOR_MAHANGHOA)]
    if indices_to_predict_mh and preprocessor_mh and model_mh and encoder_mh:
        logger.info(f"Dự đoán MaHangHoa (dependent) cho {len(indices_to_predict_mh)} bản ghi.")
        input_list_mh = []
        for i in indices_to_predict_mh:
             try:
                 input_dict = input_data.iloc[i].to_dict()
                 input_dict[config.TARGET_HACHTOAN] = results[i][config.TARGET_HACHTOAN]
                 input_list_mh.append(input_dict)
             except IndexError: results[i]["error"] = (results[i]["error"] or "") + "; Lỗi lấy data gốc cho MH"

        if input_list_mh:
            try:
                input_df_mh = pd.DataFrame(input_list_mh)
                expected_cols_mh = getattr(preprocessor_mh, 'feature_names_in_', None)
                if expected_cols_mh: input_df_mh_aligned = input_df_mh.reindex(columns=expected_cols_mh, fill_value="")
                else: input_df_mh_aligned = input_df_mh
                X_transformed_mh = preprocessor_mh.transform(input_df_mh_aligned)

                flags_od2 = [False]*len(input_df_mh); err_mh = [None]*len(input_df_mh)
                if od2: flags_od2 = check_outlier(od2, X_transformed_mh)

                pred_enc_mh = model_mh.predict(X_transformed_mh); probs_mh = model_mh.predict_proba(X_transformed_mh)
                y_pred_mh_batch = encoder_mh.inverse_transform(pred_enc_mh); prob_mh_batch = np.max(probs_mh, axis=1)

                for idx, original_index in enumerate(indices_to_predict_mh):
                    if idx < len(y_pred_mh_batch): # Kiểm tra index
                         p_mh = prob_mh_batch[idx] if prob_mh_batch[idx] is not None else None
                         if isinstance(p_mh, np.float64): p_mh = float(p_mh)
                         results[original_index][config.TARGET_MAHANGHOA] = y_pred_mh_batch[idx]
                         results[original_index][f"{config.TARGET_MAHANGHOA}_prob"] = p_mh
                         results[original_index]["is_outlier_input2"] = flags_od2[idx] if idx < len(flags_od2) else False
                         if err_mh[idx]: results[original_index]["error"] = (results[original_index]["error"] or "") + f"; Lỗi MH: {err_mh[idx]}"
                    else: logger.error(f"Index mismatch khi gộp MH: idx={idx}")

            except Exception as e:
                 logger.error(f"Client {client_id}: Lỗi khi xử lý batch MaHangHoa (dependent): {e}", exc_info=True)
                 for i in indices_to_predict_mh: results[i]["error"] = (results[i]["error"] or "") + f"; Lỗi batch MH: {e}"
    elif indices_to_predict_mh:
         logger.warning("Không thể dự đoán MH (dependent) do thiếu thành phần.")
         for i in indices_to_predict_mh: results[i]["error"] = (results[i]["error"] or "") + "; Thiếu model MH (dependent)"

    logger.info(f"Dự đoán kết hợp hoàn tất cho client {client_id}.")
    return results


# --- Hàm dự đoán chỉ HachToan ---
def predict_hachtoan_only(client_id: str, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dự đoán chỉ HachToan sử dụng preprocessor_ht."""
    logger.info(f"Bắt đầu dự đoán CHỈ HachToan cho client {client_id}.")
    components = _load_prediction_components(client_id)
    preprocessor_ht = components["preprocessor_ht"]
    model_ht = components["model_ht"]
    encoder_ht = components["encoder_ht"]
    od1 = components["outlier_detector_1"]

    if not preprocessor_ht or not encoder_ht:
        return [{"error": "Thiếu thành phần model HachToan cơ bản."}] * len(input_data)

    results_list = []
    n_items = len(input_data)
    y_pred_ht=[None]*n_items; probabilities_ht=[None]*n_items; outlier_flags_1=[False]*n_items; errors_ht=[None]*n_items
    try:
        expected_cols = getattr(preprocessor_ht, 'feature_names_in_', None)
        if expected_cols: input_data_aligned = input_data.reindex(columns=expected_cols, fill_value="")
        else: input_data_aligned = input_data
        X_transformed = preprocessor_ht.transform(input_data_aligned)
        if od1: outlier_flags_1 = check_outlier(od1, X_transformed)
        if model_ht:
            pred_enc = model_ht.predict(X_transformed); probs = model_ht.predict_proba(X_transformed)
            y_pred_ht = encoder_ht.inverse_transform(pred_enc); probabilities_ht = np.max(probs, axis=1)
        elif len(encoder_ht.classes_) == 1: y_pred_ht=[encoder_ht.classes_[0]]*n_items; probabilities_ht=[1.0]*n_items
        else: errors_ht = ["Lỗi: Model HT không tồn tại"]*n_items
    except Exception as e: errors_ht = [f"Lỗi dự đoán HachToan: {e}"]*n_items

    for i in range(n_items):
        prob = probabilities_ht[i] if i < len(probabilities_ht) and probabilities_ht[i] is not None else None
        err = errors_ht[i] if i < len(errors_ht) else None
        if isinstance(prob, np.float64): prob = float(prob)
        results_list.append({
            "HachToan": y_pred_ht[i] if i < len(y_pred_ht) else None,
            "HachToan_prob": prob,
            "is_outlier_input1": outlier_flags_1[i] if i < len(outlier_flags_1) else False,
            "error": err
        })
    return results_list


# --- Hàm dự đoán chỉ MaHangHoa (khi biết HachToan) ---
def predict_mahanghoa_only(client_id: str, input_data_with_hachtoan: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dự đoán chỉ MaHangHoa (khi biết HachToan) sử dụng preprocessor_mh."""
    logger.info(f"Bắt đầu dự đoán CHỈ MaHangHoa (dependent) cho client {client_id}.")
    components = _load_prediction_components(client_id)
    preprocessor_mh = components["preprocessor_mh"]
    model_mh = components["model_mh"]
    encoder_mh = components["encoder_mh"]
    od2 = components["outlier_detector_2"]

    if not preprocessor_mh or not encoder_mh:
        return [{"error": "Thiếu thành phần model MaHangHoa (dependent)."}] * len(input_data_with_hachtoan)
    if config.TARGET_HACHTOAN not in input_data_with_hachtoan.columns:
         return [{"error": f"Thiếu input {config.TARGET_HACHTOAN}."}] * len(input_data_with_hachtoan)

    results_list = []
    n_items = len(input_data_with_hachtoan)
    y_pred_mh=[None]*n_items; probabilities_mh=[None]*n_items; outlier_flags_2=[False]*n_items; errors=[None]*n_items
    try:
        expected_cols = getattr(preprocessor_mh, 'feature_names_in_', None)
        if expected_cols: input_data_aligned = input_data_with_hachtoan.reindex(columns=expected_cols, fill_value="")
        else: input_data_aligned = input_data_with_hachtoan
        X_transformed_mh = preprocessor_mh.transform(input_data_aligned)
        if od2: outlier_flags_2 = check_outlier(od2, X_transformed_mh)
        if model_mh:
            pred_enc = model_mh.predict(X_transformed_mh); probs = model_mh.predict_proba(X_transformed_mh)
            y_pred_mh = encoder_mh.inverse_transform(pred_enc); probabilities_mh = np.max(probs, axis=1)
        elif len(encoder_mh.classes_) == 1: y_pred_mh=[encoder_mh.classes_[0]]*n_items; probabilities_mh=[1.0]*n_items
        else: errors = ["Lỗi: Model MH (dependent) không tồn tại"]*n_items
    except Exception as e: errors = [f"Lỗi dự đoán MaHangHoa (dependent): {e}"]*n_items

    for i in range(n_items):
        prob = probabilities_mh[i] if i < len(probabilities_mh) and probabilities_mh[i] is not None else None
        err = errors[i] if i < len(errors) else None
        if isinstance(prob, np.float64): prob = float(prob)
        results_list.append({
            "MaHangHoa": y_pred_mh[i] if i < len(y_pred_mh) else None,
            "MaHangHoa_prob": prob,
            "is_outlier_input2": outlier_flags_2[i] if i < len(outlier_flags_2) else False,
            "error": err
        })
    return results_list


# --- Hàm dự đoán MaHangHoa TRỰC TIẾP (Mới) ---
def predict_mahanghoa_direct(client_id: str, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dự đoán MaHangHoa trực tiếp từ input gốc."""
    logger.info(f"Bắt đầu dự đoán MaHangHoa TRỰC TIẾP cho client {client_id}.")
    components = _load_prediction_components(client_id)
    preprocessor_ht = components["preprocessor_ht"] # Dùng preprocessor HT
    model_mh_direct = components["model_mh_direct"]
    encoder_mh = components["encoder_mh"] # Dùng encoder MH chung
    od1 = components["outlier_detector_1"] # Dùng outlier detector 1

    if not preprocessor_ht or not encoder_mh:
        return [{"error": "Thiếu thành phần preprocessor HT hoặc encoder MH."}] * len(input_data)

    results_list = []
    n_items = len(input_data)
    y_pred_mh=[None]*n_items; probabilities_mh=[None]*n_items; outlier_flags_1=[False]*n_items; errors=[None]*n_items
    try:
        expected_cols = getattr(preprocessor_ht, 'feature_names_in_', None)
        if expected_cols: input_data_aligned = input_data.reindex(columns=expected_cols, fill_value="")
        else: input_data_aligned = input_data
        X_transformed = preprocessor_ht.transform(input_data_aligned)

        if od1: outlier_flags_1 = check_outlier(od1, X_transformed)

        if model_mh_direct:
            pred_enc = model_mh_direct.predict(X_transformed); probs = model_mh_direct.predict_proba(X_transformed)
            y_pred_mh = encoder_mh.inverse_transform(pred_enc); probabilities_mh = np.max(probs, axis=1)
        elif len(encoder_mh.classes_) == 1:
             y_pred_mh=[encoder_mh.classes_[0]]*n_items; probabilities_mh=[1.0]*n_items
        else: errors = ["Lỗi: Model MH Direct không tồn tại"]*n_items
    except Exception as e: errors = [f"Lỗi dự đoán MaHangHoa Direct: {e}"]*n_items

    for i in range(n_items):
        prob = probabilities_mh[i] if i < len(probabilities_mh) and probabilities_mh[i] is not None else None
        err = errors[i] if i < len(errors) else None
        if isinstance(prob, np.float64): prob = float(prob)
        # Chỉ trả về thông tin MH và outlier của input gốc
        results_list.append({
            "MaHangHoa": y_pred_mh[i] if i < len(y_pred_mh) else None,
            "MaHangHoa_prob": prob,
            "is_outlier_input1": outlier_flags_1[i] if i < len(outlier_flags_1) else False, # Dùng flag 1
            "error": err
        })
    return results_list