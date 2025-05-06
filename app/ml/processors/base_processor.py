# app/ml/processors/base_processor.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class BaseProcessor(BaseEstimator, TransformerMixin):
    """
    Lớp cơ sở cho các processor tùy chỉnh.
    Cung cấp phương thức fit mặc định (không làm gì) và đảm bảo input/output là DataFrame.
    """
    def __init__(self, columns: list = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        # Hầu hết các preprocessor đơn giản không cần fit phức tạp,
        # trừ khi chúng học tham số từ dữ liệu (ví dụ: imputer học mean/median)
        # Hoặc trừ khi chúng là wrapper cho các transformer khác của sklearn
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Lớp con sẽ override phương thức này
        raise NotImplementedError("Phương thức transform phải được implement bởi lớp con.")

    def _check_input_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kiểm tra X là DataFrame, nếu không thì chuyển đổi."""
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, pd.Series):
                return X.to_frame()
            try:
                return pd.DataFrame(X)
            except Exception as e:
                raise TypeError(f"Input phải là DataFrame hoặc có thể chuyển đổi thành DataFrame. Lỗi: {e}")
        return X.copy() # Trả về bản copy để tránh thay đổi DataFrame gốc

    def _get_columns_to_process(self, X: pd.DataFrame) -> list:
        """
        Lấy danh sách các cột để xử lý.
        Nếu self.columns được chỉ định, dùng nó. Nếu không, dùng tất cả cột của X.
        """
        if self.columns:
            # Kiểm tra xem các cột được chỉ định có tồn tại trong X không
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Các cột được chỉ định không tìm thấy trong DataFrame: {missing_cols}")
            return self.columns
        return X.columns.tolist()