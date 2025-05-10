# app/ml/data_handler.py
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import app.core.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    # ... (Giữ nguyên) ...
    df_columns_str = df.columns.astype(str).tolist()
    missing_cols = [col for col in required_columns if col not in df_columns_str]
    if missing_cols:
        logger.error(f"Thiếu các cột bắt buộc: {missing_cols}")
        return False, missing_cols
    return True, []

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame: # Chỉ cần df làm input
    """
    Làm sạch cơ bản DataFrame.
    Xử lý NaN trong input/MaHangHoa.
    Chỉ dropna và astype cho HachToan.
    """
    target_hachtoan = config.TARGET_HACHTOAN
    target_mahanghoa = config.TARGET_MAHANGHOA
    all_columns_str = df.columns.astype(str).tolist()

    # Xử lý NaN trong các cột KHÔNG PHẢI HachToan
    for col in all_columns_str:
        if col != target_hachtoan: # Xử lý tất cả cột khác, bao gồm cả MaHangHoa nếu có
            if df[col].isnull().any():
                # Kiểm tra kiểu dữ liệu để fillna phù hợp hơn (tạm thời vẫn dùng chuỗi rỗng)
                # if pd.api.types.is_numeric_dtype(df[col]):
                #     # fill_value = df[col].median() # Hoặc mean()
                # else:
                #     fill_value = ""
                fill_value = "" # Tạm thời fill bằng chuỗi rỗng cho đơn giản
                logger.debug(f"Tìm thấy giá trị NaN trong cột '{col}', đang điền bằng '{fill_value}'.")
                df[col] = df[col].fillna(fill_value)

    # Chỉ drop rows thiếu giá trị target HachToan và astype cho HachToan
    if target_hachtoan in df.columns:
        original_len = len(df)
        # Quan trọng: dropna chỉ trên HachToan
        df.dropna(subset=[target_hachtoan], inplace=True)
        dropped_count = original_len - len(df)
        if dropped_count > 0:
            logger.warning(f"Đã loại bỏ {dropped_count} hàng do thiếu giá trị target trong cột: ['{target_hachtoan}']")

        # Đảm bảo kiểu dữ liệu string cho HachToan
        if target_hachtoan in df.columns:  # Kiểm tra lại sau dropna
            try:
                # Xử lý trường hợp có giá trị NumPy array
                if any(isinstance(x, np.ndarray) for x in df[target_hachtoan].values):
                    logger.warning(
                        f"Phát hiện giá trị NumPy array trong cột '{target_hachtoan}'. Chuyển đổi từng phần tử.")
                    df[target_hachtoan] = df[target_hachtoan].apply(
                        lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x)
                    )
                else:
                    df[target_hachtoan] = df[target_hachtoan].astype(str)
            except Exception as e:
                logger.warning(f"Không thể chuyển cột target '{target_hachtoan}' sang str: {e}.")
    else:
        # Nếu cột HachToan không tồn tại, đây là lỗi nghiêm trọng đã được check ở load_training_data
        logger.error(f"Cột target bắt buộc '{target_hachtoan}' không có trong DataFrame tại bước làm sạch.")
        # Trả về DataFrame rỗng để báo hiệu lỗi
        return pd.DataFrame()


    # Đảm bảo MaHangHoa là string nếu nó tồn tại (sau khi fillna)
    if target_mahanghoa in df.columns:
         try:
             df[target_mahanghoa] = df[target_mahanghoa].astype(str)
         except Exception as e:
              logger.warning(f"Không thể chuyển cột '{target_mahanghoa}' sang str sau fillna: {e}.")

    return df

def load_training_data(client_id: str, file_path: Path) -> Optional[pd.DataFrame]:
    """
    Tải và xử lý cơ bản dữ liệu huấn luyện từ một file CSV.
    Chỉ yêu cầu bắt buộc các cột cho model HachToan.
    Chỉ dropna trên cột HachToan.
    """
    logger.info(f"Đang tải dữ liệu huấn luyện cho client '{client_id}' từ: {file_path}")
    try:
        df = pd.read_csv(
            file_path, sep=None, engine='python',
            skipinitialspace=True, encoding='utf-8-sig'
        )

        if df.empty:
            logger.warning(f"File dữ liệu {file_path} rỗng sau khi đọc.")
            return pd.DataFrame()

        if df.shape[1] <= 1:
            logger.warning(f"File {file_path} chỉ có {df.shape[1]} cột...")

        df.columns = df.columns.astype(str)
        all_columns = df.columns.tolist()
        logger.debug(f"Các cột được đọc từ file {file_path}: {all_columns}")

        # --- Kiểm tra các cột BẮT BUỘC cho model HachToan ---
        required_hachtoan_cols = config.INPUT_COLUMNS + [config.TARGET_HACHTOAN]
        is_valid, missing_cols = _validate_columns(df, required_hachtoan_cols)
        if not is_valid:
            logger.error(f"File {file_path} thiếu cột BẮT BUỘC: {missing_cols}.")
            return pd.DataFrame()

        # --- Kiểm tra (không bắt buộc) cột MaHangHoa ---
        if config.TARGET_MAHANGHOA not in df.columns:
            logger.warning(f"File {file_path} thiếu cột '{config.TARGET_MAHANGHOA}'.")
        else:
            logger.info(f"File {file_path} có chứa cột '{config.TARGET_MAHANGHOA}'.")

        # --- Làm sạch dữ liệu (chỉ dropna trên HachToan) ---
        df_cleaned = _clean_dataframe(df.copy()) # Gọi hàm clean đã sửa

        if df_cleaned.empty:
             logger.warning(f"Không còn dữ liệu nào sau khi làm sạch (có thể do thiếu target HachToan) từ file {file_path}")
             return pd.DataFrame()

        logger.info(f"Tải thành công và làm sạch cơ bản {len(df_cleaned)} bản ghi từ {file_path}.")
        return df_cleaned

    except FileNotFoundError:
        logger.error(f"Không tìm thấy file dữ liệu: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"File dữ liệu hoàn toàn rỗng (EmptyDataError): {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đọc hoặc xử lý file {file_path}: {e}", exc_info=True)
        return None

def load_all_client_data(client_id: str) -> Optional[pd.DataFrame]:
    """Tải toàn bộ dữ liệu (initial + incremental) của một client."""
    # ... (Phần load file initial và incremental giữ nguyên) ...
    client_data_path = config.BASE_DATA_PATH / client_id
    client_data_path.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    initial_data_file = client_data_path / config.TRAINING_DATA_FILENAME
    if initial_data_file.exists():
        logger.info(f"Đang tải dữ liệu ban đầu: {initial_data_file}")
        df_initial = load_training_data(client_id, initial_data_file)
        if df_initial is not None and not df_initial.empty: all_dfs.append(df_initial)
        elif df_initial is None: logger.error(f"Lỗi nghiêm trọng khi tải file ban đầu: {initial_data_file}."); return None
        else: logger.warning(f"File dữ liệu ban đầu rỗng/không hợp lệ: {initial_data_file}")
    else:
        logger.warning(f"Không tìm thấy file dữ liệu ban đầu {initial_data_file}.")

    incremental_files = sorted(client_data_path.glob(f"{config.INCREMENTAL_DATA_PREFIX}*.csv"))
    logger.info(f"Tìm thấy {len(incremental_files)} file dữ liệu tăng cường.")
    for file in incremental_files:
        logger.info(f"Đang tải dữ liệu tăng cường: {file}")
        df_incremental = load_training_data(client_id, file)
        if df_incremental is not None and not df_incremental.empty: all_dfs.append(df_incremental)
        elif df_incremental is None: logger.warning(f"Lỗi nghiêm trọng khi tải file tăng cường: {file}. Bỏ qua.")
        else: logger.warning(f"File dữ liệu tăng cường rỗng/không hợp lệ: {file}. Bỏ qua.")

    if not all_dfs:
        logger.error(f"Không có dữ liệu hợp lệ nào được tìm thấy cho client {client_id}.")
        return None

    logger.info(f"Kết hợp {len(all_dfs)} dataframe thành một.")
    try: combined_df = pd.concat(all_dfs, ignore_index=True)
    except Exception as e: logger.error(f"Lỗi khi kết hợp DataFrame: {e}", exc_info=True); return None

    # --- Bước Deduplicate giữ nguyên ---
    initial_len = len(combined_df)
    subset_cols = [col for col in config.INPUT_COLUMNS if col in combined_df.columns]
    if subset_cols:
        try:
            combined_df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
            dedup_count = initial_len - len(combined_df)
            if dedup_count > 0: logger.info(f"Đã loại bỏ {dedup_count} bản ghi trùng lặp.")
        except Exception as e: logger.warning(f"Lỗi khi loại bỏ trùng lặp: {e}.")

    logger.info(f"Tổng số bản ghi sau khi kết hợp và loại bỏ trùng lặp: {len(combined_df)}")

    # Đảm bảo HachToan là str (nếu tồn tại)
    if config.TARGET_HACHTOAN in combined_df.columns:
        combined_df[config.TARGET_HACHTOAN] = combined_df[config.TARGET_HACHTOAN].astype(str)
    # Không cần ép kiểu MaHangHoa ở đây nữa vì _clean_dataframe đã làm (nếu cột tồn tại)

    return combined_df

# --- prepare_prediction_data giữ nguyên ---
def prepare_prediction_data(input_data: List[dict]) -> pd.DataFrame:
    # ... (code giữ nguyên) ...
    if not input_data: return pd.DataFrame()
    df = pd.DataFrame(input_data)
    for col in df.columns:
        if df[col].isnull().any():
             logger.debug(f"Tìm thấy giá trị NaN trong cột input dự đoán '{col}', đang điền bằng chuỗi rỗng.")
             df[col] = df[col].fillna("")
    return df