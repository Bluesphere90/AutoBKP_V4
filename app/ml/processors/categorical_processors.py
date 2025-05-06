# app/ml/processors/categorical_processors.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from typing import Optional, List, Any, Dict # <-- Thêm Optional và các kiểu khác nếu cần

from .base_processor import BaseProcessor

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None,
                 strategy: str = 'onehot',
                 handle_unknown: str = 'ignore',
                 min_frequency: Optional[int] = None,
                 max_categories: Optional[int] = None,
                 sparse_output: bool = False):
        # --- Gán trực tiếp, không sửa đổi ---
        self.columns = columns
        self.strategy = strategy # Gán trực tiếp
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.sparse_output = sparse_output
        # ------------------------------------
        self.encoder_ = None
        self.fitted_columns_ = None
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_fit = X.copy()
        if self.columns:
            self.fitted_columns_ = [col for col in self.columns if col in X_fit.columns]
        else:
            self.fitted_columns_ = X_fit.select_dtypes(include=['object', 'category']).columns.tolist()

        if not self.fitted_columns_:
            print("CẢNH BÁO: Không có cột nào được chọn để mã hóa categorical.")
            return self

        # Chuẩn hóa strategy thành lower case KHI SỬ DỤNG
        strategy_lower = self.strategy.lower()

        if strategy_lower == 'onehot':
            self.encoder_ = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                min_frequency=self.min_frequency,
                max_categories=self.max_categories,
                sparse_output=self.sparse_output
            )
            try:
                self.encoder_.fit(X_fit[self.fitted_columns_].astype(str))
                self.feature_names_out_ = self.encoder_.get_feature_names_out(self.fitted_columns_)
            except Exception as e:
                 print(f"Lỗi khi fit OneHotEncoder trên cột {self.fitted_columns_}: {e}. Thử lại không ép kiểu.")
                 try:
                    self.encoder_.fit(X_fit[self.fitted_columns_])
                    self.feature_names_out_ = self.encoder_.get_feature_names_out(self.fitted_columns_)
                 except Exception as e2:
                      raise ValueError(f"Không thể fit OneHotEncoder: {e2}")
        else:
            # Sử dụng self.strategy gốc trong thông báo lỗi
            raise ValueError(f"Chiến lược mã hóa không hợp lệ: {self.strategy}. Hiện chỉ hỗ trợ 'onehot'.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder_ is None or self.fitted_columns_ is None:
            raise RuntimeError("CategoricalEncoder phải được fit trước khi transform.")

        X_transformed_df = X.copy()
        cols_to_transform = [col for col in self.fitted_columns_ if col in X_transformed_df.columns]

        if not cols_to_transform:
            print("CẢNH BÁO: Không có cột nào (đã fit) để transform categorical.")
            return X.drop(columns=self.fitted_columns_, errors='ignore')

        # Chuẩn hóa strategy thành lower case KHI SỬ DỤNG
        strategy_lower = self.strategy.lower()

        if strategy_lower == 'onehot':
            try:
                transformed_data = self.encoder_.transform(X_transformed_df[cols_to_transform].astype(str))
            except Exception:
                 transformed_data = self.encoder_.transform(X_transformed_df[cols_to_transform])

            if self.sparse_output:
                 # Trả về sparse matrix nếu được yêu cầu rõ ràng
                 print("CategoricalEncoder trả về sparse matrix.")
                 # Pipeline Builder hoặc ColumnTransformer sẽ xử lý tiếp
                 # Lưu ý: Việc lấy feature names out cho sparse có thể cần xử lý riêng
                 return transformed_data
            else: # Trả về DataFrame nếu sparse_output=False
                if hasattr(transformed_data, "toarray"):
                     transformed_df = pd.DataFrame(transformed_data.toarray(), columns=self.feature_names_out_, index=X.index)
                else:
                     transformed_df = pd.DataFrame(transformed_data, columns=self.feature_names_out_, index=X.index)

                X_transformed_df = X_transformed_df.drop(columns=cols_to_transform, errors='ignore')
                X_transformed_df = pd.concat([X_transformed_df, transformed_df], axis=1)
        # else: # Xử lý các strategy khác nếu có
        #     pass

        return X_transformed_df

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, 'feature_names_out_') and self.feature_names_out_ is not None:
            return self.feature_names_out_
        # Cung cấp fallback nếu cần
        if input_features and self.fitted_columns_:
             # Tính toán tên dự kiến nếu có thể (ví dụ cho onehot)
             # Điều này có thể phức tạp, trả về list rỗng là an toàn hơn
             pass
        return []


class HashingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list = None, n_features: int = 1024, alternate_sign: bool = True):
        self.columns = columns
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.hasher_ = None
        self.fitted_columns_ = None
        self.feature_names_out_ = None


    def fit(self, X: pd.DataFrame, y=None):
        X_fit = X.copy()
        if self.columns:
            self.fitted_columns_ = [col for col in self.columns if col in X_fit.columns]
        else:
            self.fitted_columns_ = X_fit.select_dtypes(include=['object', 'category']).columns.tolist()

        if not self.fitted_columns_:
            print("CẢNH BÁO: HashingEncoder không có cột nào được chọn.")
            return self

        # FeatureHasher không cần fit dữ liệu, chỉ cần cấu hình
        self.hasher_ = FeatureHasher(
            n_features=self.n_features,
            input_type='string', # Hoặc 'dict' nếu input là list of dicts
            alternate_sign=self.alternate_sign
        )
        # Tạo tên feature output
        self.feature_names_out_ = [f"hashed_feature_{i}" for i in range(self.n_features)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.hasher_ is None or self.fitted_columns_ is None:
            raise RuntimeError("HashingEncoder phải được fit (cấu hình) trước khi transform.")

        X_transformed_df = X.copy()
        cols_to_transform = [col for col in self.fitted_columns_ if col in X_transformed_df.columns]

        if not cols_to_transform:
            print("CẢNH BÁO: HashingEncoder không có cột nào (đã fit) để transform.")
            return X.drop(columns=self.fitted_columns_, errors='ignore')

        # FeatureHasher nhận input là list các dictionary hoặc list các (feature, value) tuples
        # Chúng ta cần chuyển đổi các cột được chọn thành định dạng này
        raw_X = []
        for _, row in X_transformed_df[cols_to_transform].astype(str).iterrows():
            # raw_X.append(row.to_dict().items()) # list of (feature, value) tuples
            # Hoặc list of dicts, mỗi dict cho một cột
            # Hoặc list of strings nếu FeatureHasher được cấu hình input_type='string'
            # và chúng ta truyền từng giá trị string vào
            # Tuy nhiên, FeatureHasher thường mong đợi input là list các feature-value pairs
            # cho mỗi mẫu.
            # Cách tiếp cận đơn giản hơn: áp dụng cho từng cột và nối lại, nhưng không tối ưu.
            # Cách tốt hơn: tạo input dạng list of list of strings
             current_sample_features = []
             for col_name in cols_to_transform:
                 current_sample_features.append(f"{col_name}={row[col_name]}") # Tạo feature dạng "col=value"
             raw_X.append(current_sample_features)


        # Hoặc cách khác, nếu muốn truyền từng giá trị string và để hasher xử lý:
        # raw_X_dict_list = X_transformed_df[cols_to_transform].astype(str).to_dict(orient='records')
        # transformed_data = self.hasher_.transform(raw_X_dict_list)

        # Sử dụng FeatureHasher với input_type='string' và truyền các chuỗi "column_name=value"
        # Điều này giúp FeatureHasher phân biệt giá trị từ các cột khác nhau
        transformed_data = self.hasher_.transform(raw_X)


        hashed_df = pd.DataFrame.sparse.from_spmatrix(transformed_data, columns=self.feature_names_out_, index=X.index)

        X_transformed_df = X_transformed_df.drop(columns=cols_to_transform, errors='ignore')
        X_transformed_df = pd.concat([X_transformed_df, hashed_df], axis=1)

        return X_transformed_df

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        return []