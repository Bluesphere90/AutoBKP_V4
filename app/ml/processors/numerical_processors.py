# app/ml/processors/numerical_processors.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .base_processor import BaseProcessor

class NumericalImputer(BaseProcessor):
    def __init__(self, columns: list = None, strategy: str = 'median', fill_value=None):
        super().__init__(columns)
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputers_ = {} # Lưu các imputer đã fit cho từng cột

    def fit(self, X: pd.DataFrame, y=None):
        X_fit = self._check_input_df(X)
        cols_to_process = self._get_columns_to_process(X_fit)

        for col in cols_to_process:
            if col not in X_fit.columns: continue
            # Kiểm tra cột có phải là số không
            if not pd.api.types.is_numeric_dtype(X_fit[col]):
                print(f"CẢNH BÁO: Cột '{col}' không phải kiểu số. NumericalImputer sẽ bỏ qua cột này.")
                continue

            imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
            # Fit imputer trên từng cột riêng lẻ
            # SimpleImputer cần input là 2D array (DataFrame với 1 cột)
            imputer.fit(X_fit[[col]])
            self.imputers_[col] = imputer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self._check_input_df(X)
        # Không gọi _get_columns_to_process ở đây, chỉ xử lý các cột đã fit
        for col, imputer in self.imputers_.items():
            if col in X_transformed.columns:
                if not pd.api.types.is_numeric_dtype(X_transformed[col]):
                     print(f"CẢNH BÁO (transform): Cột '{col}' không phải kiểu số. Bỏ qua imputation.")
                     continue
                # Transform trả về numpy array, cần gán lại vào DataFrame
                X_transformed[col] = imputer.transform(X_transformed[[col]])
            else:
                 print(f"CẢNH BÁO: Cột '{col}' (đã fit imputer) không tìm thấy trong DataFrame khi transform. Bỏ qua.")
        return X_transformed


class NumericalScaler(BaseProcessor):
    def __init__(self, columns: list = None, scaler_type: str = 'standard'): # 'standard' hoặc 'minmax'
        super().__init__(columns)
        self.scaler_type = scaler_type.lower()
        self.scalers_ = {} # Lưu các scaler đã fit

    def fit(self, X: pd.DataFrame, y=None):
        X_fit = self._check_input_df(X)
        cols_to_process = self._get_columns_to_process(X_fit)

        for col in cols_to_process:
            if col not in X_fit.columns: continue
            if not pd.api.types.is_numeric_dtype(X_fit[col]):
                print(f"CẢNH BÁO: Cột '{col}' không phải kiểu số. NumericalScaler sẽ bỏ qua cột này.")
                continue

            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Loại scaler không hợp lệ: {self.scaler_type}. Chỉ hỗ trợ 'standard' hoặc 'minmax'.")

            scaler.fit(X_fit[[col]])
            self.scalers_[col] = scaler
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self._check_input_df(X)
        for col, scaler in self.scalers_.items():
            if col in X_transformed.columns:
                if not pd.api.types.is_numeric_dtype(X_transformed[col]):
                     print(f"CẢNH BÁO (transform): Cột '{col}' không phải kiểu số. Bỏ qua scaling.")
                     continue
                X_transformed[col] = scaler.transform(X_transformed[[col]])
            else:
                print(f"CẢNH BÁO: Cột '{col}' (đã fit scaler) không tìm thấy trong DataFrame khi transform. Bỏ qua.")
        return X_transformed