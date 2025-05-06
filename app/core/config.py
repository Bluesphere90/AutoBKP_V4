# app/core/config.py

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum # Thêm Enum

# Load environment variables from .env file if it exists
load_dotenv()
logger = logging.getLogger(__name__) # Thêm logger

# --- Paths ---
BASE_DATA_PATH = Path(os.getenv("DATA_PATH", "/app/data/client_data"))
BASE_MODELS_PATH = Path(os.getenv("MODELS_PATH", "/app/models/client_models"))
BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)
BASE_MODELS_PATH.mkdir(parents=True, exist_ok=True)

# --- Filenames ---
TRAINING_DATA_FILENAME = "training_data.csv"
INCREMENTAL_DATA_PREFIX = "incremental_data_"
METADATA_FILENAME_PREFIX = "metadata_" # Thêm prefix cho metadata
# Tên file cố định cho model/preprocessor/outlier/encoder
PREPROCESSOR_HACHTOAN_FILENAME = "preprocessor_hachtoan.joblib"
PREPROCESSOR_MAHANGHOA_FILENAME = "preprocessor_mahanghoa.joblib"
HACHTOAN_MODEL_FILENAME = "hachtoan_model.joblib"
MAHANGHOA_MODEL_FILENAME = "mahanghoa_model.joblib" # Model MH dựa trên HachToan=15*
MAHANGHOA_DIRECT_MODEL_FILENAME = "mahanghoa_direct_model.joblib" # <-- Tên file model MH mới
HACHTOAN_ENCODER_FILENAME = "hachtoan_encoder.joblib"
MAHANGHOA_ENCODER_FILENAME = "mahanghoa_encoder.joblib" # Dùng chung encoder cho cả 2 model MH
OUTLIER_DETECTOR_1_FILENAME = "outlier_detector_1.joblib" # Dùng cho input của model HT và MH Direct
OUTLIER_DETECTOR_2_FILENAME = "outlier_detector_2.joblib" # Dùng cho input của model MH (có HachToan)
# OUTLIER_DETECTOR_3_FILENAME = "outlier_detector_3.joblib" # Có thể tạo riêng nếu cần, tạm dùng chung OD1
LABEL_ENCODERS_DIR = "label_encoders"

# --- Column Names ---
INPUT_COLUMNS = ["MSTNguoiBan", "TenHangHoaDichVu"]
TARGET_HACHTOAN = "HachToan"
TARGET_MAHANGHOA = "MaHangHoa"

# --- Model Logic ---
HACHTOAN_PREFIX_FOR_MAHANGHOA = "15"

# --- Model Selection & Parameters ---
# Sử dụng Enum để định nghĩa các loại model hỗ trợ
class SupportedModels(str, Enum):
    RANDOM_FOREST = "RandomForestClassifier"
    LOGISTIC_REGRESSION = "LogisticRegression"
    MULTINOMIAL_NB = "MultinomialNB"
    LINEAR_SVC = "LinearSVC"
    # Thêm các model khác nếu muốn hỗ trợ, ví dụ:
    # LIGHTGBM = "LGBMClassifier"

# Tham số mặc định cho từng loại model
DEFAULT_MODEL_PARAMS = {
    SupportedModels.RANDOM_FOREST: {
        "n_estimators": 100, "random_state": 42, "class_weight": "balanced",
        "n_jobs": -1, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1
    },
    SupportedModels.LOGISTIC_REGRESSION: {
        "multi_class": 'ovr', "solver": 'liblinear', "random_state": 42,
        "class_weight": 'balanced', "max_iter": 1000, "C": 1.0
    },
    SupportedModels.MULTINOMIAL_NB: {
        "alpha": 1.0 # Tham số smoothing
    },
    SupportedModels.LINEAR_SVC: {
        "C": 1.0, "random_state": 42, "class_weight": "balanced", "max_iter": 2000,
        "dual": "auto" # Thêm dual='auto' để tránh warning
    }
    # Thêm params cho các model khác
    # SupportedModels.LIGHTGBM: {"n_estimators": 100, ...}
}

# Đọc cấu hình model từ biến môi trường (ưu tiên hơn giá trị mặc định)
# Biến này chỉ dùng làm mặc định nếu không có gì được chọn/lưu
DEFAULT_MODEL_TYPE = SupportedModels.RANDOM_FOREST # Chọn RF làm mặc định nếu không có gì khác
SELECTED_MODEL_NAME_FROM_ENV = os.getenv("SELECTED_MODEL_NAME", DEFAULT_MODEL_TYPE.value)

# --- Other ML Params ---
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
OUTLIER_CONTAMINATION_STR = os.getenv("OUTLIER_CONTAMINATION", 'auto')
try:
    # Cố gắng chuyển đổi sang float nếu là số
    OUTLIER_CONTAMINATION = float(OUTLIER_CONTAMINATION_STR)
except ValueError:
    # Nếu không phải số (ví dụ: 'auto'), giữ nguyên dạng string
    OUTLIER_CONTAMINATION = OUTLIER_CONTAMINATION_STR

VALIDATION_SET_SIZE = float(os.getenv("VALIDATION_SET_SIZE", 0.2)) # Thêm cấu hình tỷ lệ validation

# --- API Configuration ---
API_TITLE = "HachToan & MaHangHoa Prediction API"
API_DESCRIPTION = "API for training and predicting HachToan and MaHangHoa using ML."
API_VERSION = "0.1.0"

# --- Column Configuration Enums ---
# Định nghĩa các loại cột được hỗ trợ
class SupportedColumnType(str, Enum):
    TEXT = "text"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DATETIME = "datetime" # Ví dụ thêm datetime
    IGNORE = "ignore"     # Để bỏ qua cột

# Định nghĩa các ngôn ngữ được hỗ trợ (cho xử lý text)
class SupportedLanguage(str, Enum):
    VIETNAMESE = "vi"
    ENGLISH = "en"
    NONE = "none" # Cho text không cần xử lý ngôn ngữ cụ thể

# Định nghĩa các chiến lược mã hóa categorical
class CategoricalStrategy(str, Enum):
    ONEHOT = "onehot"
    HASHING = "hashing"
    # TARGET = "target" # Ví dụ thêm target encoding

# Định nghĩa các chiến lược xử lý số
class NumericalStrategy(str, Enum):
    STANDARD_SCALER = "standard"
    MINMAX_SCALER = "minmax"
    MEDIAN_IMPUTER = "median"
    MEAN_IMPUTER = "mean"

# Định nghĩa các chiến lược vector hóa text
class TextVectorizerStrategy(str, Enum):
    TFIDF = "tfidf"
    # HASHING = "hashing" # Có thể dùng HashingVectorizer cho text
    # EMBEDDING = "embedding" # Ví dụ
