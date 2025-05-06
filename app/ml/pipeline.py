# app/ml/pipeline.py

import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer # Có thể cần cho các bước tùy chỉnh

# Import các lớp processor đã định nghĩa
from app.ml.processors import (
    VietnameseTextCleaner,
    VietnameseWordTokenizer,
    EnglishTextCleaner,
    EnglishWordTokenizer,
    StopwordRemover,
    TfidfVectorizerWrapper,
    NumericalImputer,
    NumericalScaler,
    CategoricalEncoder,
    HashingEncoder
)
# Import config để lấy các giá trị mặc định nếu cần
import app.core.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Processor Registry ---
# Ánh xạ từ (type, subtype/language) hoặc type đến lớp processor tương ứng
PROCESSOR_REGISTRY = {
    # Text Processors
    ("text", "vi", "clean"): VietnameseTextCleaner,
    ("text", "en", "clean"): EnglishTextCleaner, # Giả sử có EnglishTextCleaner
    ("text", "vi", "tokenize"): VietnameseWordTokenizer,
    ("text", "en", "tokenize"): EnglishWordTokenizer, # Giả sử có
    ("text", "vi", "stopwords"): lambda params: StopwordRemover(language='vi', **params), # Dùng lambda nếu cần truyền tham số đặc biệt
    ("text", "en", "stopwords"): lambda params: StopwordRemover(language='en', **params),
    ("text", None, "tfidf"): TfidfVectorizerWrapper, # Xử lý TFIDF chung (cần cột đã tiền xử lý)

    # Numerical Processors
    "numerical_imputer": NumericalImputer,
    "numerical_scaler": NumericalScaler,

    # Categorical Processors
    "categorical_onehot": CategoricalEncoder, # Mặc định strategy='onehot'
    "categorical_hashing": HashingEncoder,

    # Có thể thêm các loại khác: datetime, ignore, etc.
}

# --- Dynamic Pipeline Builder ---

def build_dynamic_preprocessor(column_metadata: Dict[str, Dict[str, Any]]) -> ColumnTransformer:
    """
    Xây dựng ColumnTransformer động dựa trên metadata cấu hình cột.

    Args:
        column_metadata: Dictionary chứa cấu hình cho từng cột.
                         Ví dụ: {"columns": {"col_name": {"type": "text", "language": "vi", ...}}}

    Returns:
        Một đối tượng ColumnTransformer chưa được fit.
    """
    transformers = [] # List các tuple ('name', transformer_pipeline, columns)
    processed_columns = set() # Theo dõi các cột đã được đưa vào transformer

    if not column_metadata or "columns" not in column_metadata:
        raise ValueError("Metadata cấu hình cột không hợp lệ hoặc thiếu key 'columns'.")

    metadata_cols = column_metadata["columns"]

    # --- Xử lý các cột TEXT ---
    text_cols_config = {
        col: cfg for col, cfg in metadata_cols.items()
        if cfg.get("type") == "text" and cfg.get("language") # Chỉ xử lý text có language
    }
    # Nhóm các cột text theo ngôn ngữ và các bước xử lý
    text_pipelines_by_lang: Dict[Tuple[str, str], Tuple[Pipeline, List[str]]] = {}

    for col, cfg in text_cols_config.items():
        lang = cfg.get("language")
        pipeline_steps = []
        params = cfg.get("params", {}) # Lấy các tham số tùy chỉnh nếu có

        # 1. Cleaning
        cleaner_key = ("text", lang, "clean")
        if cleaner_key in PROCESSOR_REGISTRY:
            cleaner_params = params.get("cleaner", {}) # Tham số riêng cho cleaner
            pipeline_steps.append(('clean', PROCESSOR_REGISTRY[cleaner_key](**cleaner_params)))

        # 2. Tokenization
        tokenizer_key = ("text", lang, "tokenize")
        if tokenizer_key in PROCESSOR_REGISTRY:
             tokenizer_params = params.get("tokenizer", {})
             pipeline_steps.append(('tokenize', PROCESSOR_REGISTRY[tokenizer_key](**tokenizer_params)))

        # 3. Stopwords (tùy chọn)
        if cfg.get("remove_stopwords", False): # Cần key 'remove_stopwords' trong metadata
             stopwords_key = ("text", lang, "stopwords")
             if stopwords_key in PROCESSOR_REGISTRY:
                 stopwords_params = params.get("stopwords", {})
                 # Truyền custom_stopwords từ metadata nếu có
                 custom_list = cfg.get("custom_stopwords")
                 if custom_list: stopwords_params["custom_stopwords"] = custom_list
                 processor_func = PROCESSOR_REGISTRY[stopwords_key]
                 pipeline_steps.append(('stopwords', processor_func(stopwords_params)))

        # 4. Vectorization (TF-IDF hoặc khác)
        vectorizer_type = cfg.get("vectorizer", "tfidf") # Mặc định là tfidf
        vectorizer_key = ("text", None, vectorizer_type) # Key chung cho vectorizer
        if vectorizer_key in PROCESSOR_REGISTRY:
            vectorizer_params = params.get("vectorizer", {})
            # Truyền các tham số TFIDF từ metadata (ví dụ: max_features, ngram_range)
            tfidf_cfg = {k:v for k,v in cfg.items() if k in ['max_features', 'ngram_range', 'min_df', 'max_df']}

            # Đảm bảo giá trị 'ngram_range' trong tfidf_cfg là tuple nếu nó tồn tại
            if 'ngram_range' in tfidf_cfg and isinstance(tfidf_cfg['ngram_range'], list):
                logger.warning(f"Chuyển đổi 'ngram_range' từ list sang tuple cho cột '{col}'.")
                tfidf_cfg['ngram_range'] = tuple(tfidf_cfg['ngram_range'])
            # ----------------------

            vectorizer_params.update(tfidf_cfg)
            # TfidfVectorizerWrapper cần được fit trên cột của nó, không phải list cột
            # Do đó, không thể dùng trực tiếp trong pipeline này nếu pipeline áp dụng cho nhiều cột
            # Cần tách riêng bước vectorizer ra khỏi pipeline xử lý text chung?
            # HOẶC: Thiết kế lại TfidfVectorizerWrapper để nhận pipeline text làm input?

            # --> Cách tiếp cận đơn giản hơn: Mỗi cột text có pipeline riêng đến vectorizer
            # Tạo pipeline hoàn chỉnh cho cột text này
            text_pipeline_for_col = Pipeline(steps=pipeline_steps + [
                ('vectorize', PROCESSOR_REGISTRY[vectorizer_key](vectorizer_params=vectorizer_params))
            ])
            # Thêm vào transformers chính, áp dụng cho cột này
            transformers.append((f"text_{lang}_{col}", text_pipeline_for_col, [col]))
            processed_columns.add(col)

        else:
            # Nếu chỉ có các bước tiền xử lý text mà không có vectorizer
            # Có thể tạo pipeline chỉ chứa các bước đó nếu cần output text đã xử lý
            if pipeline_steps:
                 text_preprocessing_pipeline = Pipeline(steps=pipeline_steps)
                 transformers.append((f"text_preprocess_{lang}_{col}", text_preprocessing_pipeline, [col]))
                 processed_columns.add(col)


    # --- Xử lý các cột NUMERICAL ---
    numerical_cols_config = {
        col: cfg for col, cfg in metadata_cols.items() if cfg.get("type") == "numerical"
    }
    numerical_cols = list(numerical_cols_config.keys())
    if numerical_cols:
        num_pipeline_steps = []
        # 1. Imputation (tùy chọn)
        # Kiểm tra xem có cột nào cần impute không (hoặc luôn thêm nếu có config)
        impute_strategy = column_metadata.get("default_numerical_imputer", "median") # Lấy default từ root metadata
        # Hoặc kiểm tra từng cột config
        needs_impute = any(cfg.get("imputer_strategy") for cfg in numerical_cols_config.values())
        if needs_impute or True: # Tạm thời luôn thêm imputer nếu có cột số
             imputer_key = "numerical_imputer"
             if imputer_key in PROCESSOR_REGISTRY:
                 # Có thể cho phép override strategy cho từng cột, nhưng phức tạp hơn
                 # Tạm dùng strategy chung
                 imputer_params = {"strategy": impute_strategy}
                 num_pipeline_steps.append(('imputer', PROCESSOR_REGISTRY[imputer_key](**imputer_params)))

        # 2. Scaling (tùy chọn)
        scaler_type = column_metadata.get("default_numerical_scaler", "standard")
        needs_scale = any(cfg.get("scaler") for cfg in numerical_cols_config.values())
        if needs_scale or True: # Tạm thời luôn thêm scaler
            scaler_key = "numerical_scaler"
            if scaler_key in PROCESSOR_REGISTRY:
                 scaler_params = {"scaler_type": scaler_type}
                 num_pipeline_steps.append(('scaler', PROCESSOR_REGISTRY[scaler_key](**scaler_params)))

        if num_pipeline_steps:
            numerical_pipeline = Pipeline(steps=num_pipeline_steps)
            transformers.append(('numerical_processing', numerical_pipeline, numerical_cols))
            processed_columns.update(numerical_cols)


    # --- Xử lý các cột CATEGORICAL ---
    categorical_cols_config = {
        col: cfg for col, cfg in metadata_cols.items() if cfg.get("type") == "categorical"
    }
    # Nhóm theo chiến lược mã hóa
    ohe_cols = [col for col, cfg in categorical_cols_config.items() if cfg.get("strategy", "onehot") == "onehot"]
    hash_cols = [col for col, cfg in categorical_cols_config.items() if cfg.get("strategy") == "hashing"]

    if ohe_cols:
        ohe_key = "categorical_onehot"
        if ohe_key in PROCESSOR_REGISTRY:
            # Lấy tham số chung hoặc cho từng cột (hiện tại dùng chung)
            ohe_params = {
                "handle_unknown": column_metadata.get("default_categorical_handle_unknown", "ignore"),
                "sparse_output": False # Đặt False để dễ dàng làm việc với DataFrame output
            }
            # Có thể đọc thêm min_frequency, max_categories từ metadata
            transformers.append(('categorical_onehot', PROCESSOR_REGISTRY[ohe_key](**ohe_params), ohe_cols))
            processed_columns.update(ohe_cols)

    if hash_cols:
        hash_key = "categorical_hashing"
        if hash_key in PROCESSOR_REGISTRY:
             # Lấy tham số chung hoặc cho từng cột
             hash_params = {
                 "n_features": column_metadata.get("default_hashing_n_features", 1024)
             }
             transformers.append(('categorical_hashing', PROCESSOR_REGISTRY[hash_key](**hash_params), hash_cols))
             processed_columns.update(hash_cols)

    # --- Xử lý các loại cột khác (ví dụ: datetime, ignore) ---
    # ... (Thêm logic tương tự cho các type khác nếu cần) ...
    ignored_cols = [col for col, cfg in metadata_cols.items() if cfg.get("type") == "ignore"]
    processed_columns.update(ignored_cols) # Đánh dấu là đã xử lý (bằng cách bỏ qua)


    # --- Xây dựng ColumnTransformer cuối cùng ---
    # remainder='passthrough' để giữ lại các cột không được định nghĩa trong metadata
    # hoặc 'drop' nếu muốn loại bỏ chúng
    remainder_strategy = column_metadata.get("remainder_strategy", "passthrough")

    # Kiểm tra xem có transformer nào được tạo không
    if not transformers:
        logger.warning("Không có transformers nào được tạo từ metadata. Preprocessor sẽ không làm gì.")
        # Trả về một transformer 'rỗng' nếu không có gì để xử lý
        # Hoặc có thể raise lỗi tùy theo yêu cầu
        # Sử dụng FunctionTransformer không làm gì cả
        def no_op(X): return X
        return FunctionTransformer(no_op, validate=False)


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder_strategy,
        verbose_feature_names_out=False # Tránh tên cột dài dòng từ ColumnTransformer
    )
    # preprocessor.set_output(transform="pandas") # Yêu cầu output là DataFrame (sklearn >= 1.2)

    logger.info(f"Đã xây dựng preprocessor động với {len(transformers)} bộ xử lý.")
    logger.debug(f"Các cột đã được xử lý: {processed_columns}")
    logger.debug(f"Chiến lược remainder: {remainder_strategy}")

    return preprocessor