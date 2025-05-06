# app/ml/processors/text_processors.py

import re
import logging # Thêm logging
import pandas as pd
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Import từ base_processor
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__) # Khởi tạo logger

# --- Vietnamese NLP Library Handling ---
VI_TOKENIZER_FUNC = None
try:
    from underthesea import word_tokenize as underthesea_tokenize
    VI_TOKENIZER_FUNC = lambda text: underthesea_tokenize(text, format="text")
    logger.info("Sử dụng underthesea để tách từ tiếng Việt.")
    VI_TOKENIZER_AVAILABLE = True
except ImportError:
    logger.warning("Không tìm thấy thư viện 'underthesea'. Đang thử 'pyvi'...")
    try:
        from pyvi import ViTokenizer
        VI_TOKENIZER_FUNC = ViTokenizer.tokenize
        logger.info("Sử dụng pyvi để tách từ tiếng Việt.")
        VI_TOKENIZER_AVAILABLE = True
    except ImportError:
        logger.warning("Không tìm thấy 'pyvi'. Sẽ sử dụng tách từ cơ bản bằng khoảng trắng cho tiếng Việt.")
        VI_TOKENIZER_AVAILABLE = False
        # Fallback tokenizer
        def fallback_tokenize(text):
            return ' '.join(text.split())
        VI_TOKENIZER_FUNC = fallback_tokenize

# --- English NLP Library Handling (Ví dụ với NLTK - cần cài: pip install nltk) ---
EN_TOKENIZER_FUNC = None
try:
    import nltk
    # Download punkt nếu chưa có: nltk.download('punkt') trong môi trường Python
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        logger.info("Downloading NLTK 'punkt' tokenizer models...")
        nltk.download('punkt', quiet=True) # Tải lặng lẽ

    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    EN_TOKENIZER_FUNC = lambda text: " ".join(nltk_word_tokenize(text))
    logger.info("Sử dụng nltk.word_tokenize để tách từ tiếng Anh.")
    EN_TOKENIZER_AVAILABLE = True
except ImportError:
    logger.warning("Không tìm thấy thư viện 'nltk'. Sẽ sử dụng tách từ cơ bản bằng regex cho tiếng Anh.")
    EN_TOKENIZER_AVAILABLE = False
    # Fallback tokenizer cho tiếng Anh
    def fallback_en_tokenize(text):
        return ' '.join(re.findall(r'\b\w+\b', text.lower()))
    EN_TOKENIZER_FUNC = fallback_en_tokenize


# --- Base Text Cleaner (Có thể dùng chung logic) ---
class BaseTextCleaner(BaseProcessor):
    def __init__(self, columns: list = None, to_lower: bool = True, remove_punctuation: bool = True, remove_digits: bool = False, punctuation_regex: str = r'[^\w\s]'):
        super().__init__(columns)
        self.to_lower = to_lower
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.punctuation_regex = punctuation_regex

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self._check_input_df(X)
        cols_to_process = self._get_columns_to_process(X_transformed)

        for col in cols_to_process:
            if col not in X_transformed.columns: continue
            series = X_transformed[col].astype(str)
            if self.to_lower: series = series.str.lower()
            if self.remove_punctuation: series = series.apply(lambda x: re.sub(self.punctuation_regex, '', x))
            if self.remove_digits: series = series.apply(lambda x: re.sub(r'\d+', '', x))
            series = series.apply(lambda x: ' '.join(x.split())) # Remove extra whitespace
            X_transformed[col] = series
        return X_transformed

# --- Specific Cleaners ---
class VietnameseTextCleaner(BaseTextCleaner):
    def __init__(self, columns: list = None, to_lower: bool = True, remove_punctuation: bool = True, remove_digits: bool = False):
        # Regex giữ lại ký tự tiếng Việt
        vi_punctuation_regex = r'[^\w\s]' # \w bao gồm cả ký tự tiếng Việt có dấu trong Python 3
        super().__init__(columns, to_lower, remove_punctuation, remove_digits, vi_punctuation_regex)
        logger.debug("Khởi tạo VietnameseTextCleaner")

class EnglishTextCleaner(BaseTextCleaner):
    def __init__(self, columns: list = None, to_lower: bool = True, remove_punctuation: bool = True, remove_digits: bool = False):
        # Regex cho tiếng Anh (có thể giống hoặc khác tùy yêu cầu)
        en_punctuation_regex = r'[^\w\s]'
        super().__init__(columns, to_lower, remove_punctuation, remove_digits, en_punctuation_regex)
        logger.debug("Khởi tạo EnglishTextCleaner")

# --- Word Tokenizers ---
class VietnameseWordTokenizer(BaseProcessor):
    def __init__(self, columns: list = None):
        super().__init__(columns)
        if not VI_TOKENIZER_AVAILABLE:
            logger.warning("VietnameseWordTokenizer được sử dụng nhưng thư viện NLP không khả dụng.")
        self.tokenizer = VI_TOKENIZER_FUNC # Sử dụng hàm đã chọn ở trên
        logger.debug("Khởi tạo VietnameseWordTokenizer")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self._check_input_df(X)
        cols_to_process = self._get_columns_to_process(X_transformed)
        for col in cols_to_process:
            if col not in X_transformed.columns: continue
            series = X_transformed[col].astype(str)
            X_transformed[col] = series.apply(self.tokenizer)
        return X_transformed

class EnglishWordTokenizer(BaseProcessor):
    def __init__(self, columns: list = None):
        super().__init__(columns)
        self.tokenizer = EN_TOKENIZER_FUNC # Sử dụng hàm đã chọn ở trên
        logger.debug("Khởi tạo EnglishWordTokenizer")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self._check_input_df(X)
        cols_to_process = self._get_columns_to_process(X_transformed)
        for col in cols_to_process:
            if col not in X_transformed.columns: continue
            series = X_transformed[col].astype(str)
            X_transformed[col] = series.apply(self.tokenizer)
        return X_transformed


# --- Stopword Remover ---
class StopwordRemover(BaseProcessor):
    def __init__(self, columns: list = None, language: str = 'vi', custom_stopwords: list = None):
        super().__init__(columns)
        self.language = language.lower()
        self.stopwords = self._load_stopwords(custom_stopwords)
        logger.debug(f"Khởi tạo StopwordRemover cho ngôn ngữ '{self.language}' với {len(self.stopwords)} stopwords.")

    def _load_stopwords(self, custom_stopwords: list = None) -> set:
        stopwords = set()
        # --- Cần có cơ chế tải stopwords tốt hơn ---
        # Ví dụ: từ file hoặc thư viện chuẩn
        if self.language == 'vi':
            # Ví dụ stopwords cơ bản
            stopwords.update(["bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa", "có", "có thể", "cứ", "cùng", "cũng", "đã", "đang", "đây", "để", "đến", "đều", "điều", "do", "đó", "được", "dưới", "gì", "khi", "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "một", "nên", "nếu", "ngay", "ngoài", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng", "và", "vẫn", "vào", "vậy", "vì", "việc", "với", "vừa"])
            logger.debug("Đã tải stopwords tiếng Việt cơ bản.")
        elif self.language == 'en':
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                # nltk.download('stopwords') # Đảm bảo đã tải
                stopwords.update(nltk_stopwords.words('english'))
                logger.debug("Đã tải stopwords tiếng Anh từ NLTK.")
            except ImportError:
                logger.warning("Không tìm thấy NLTK để tải stopwords tiếng Anh. Sử dụng danh sách cơ bản.")
                stopwords.update(["a", "an", "the", "in", "on", "at", "is", "are", "was", "were", "to", "of", "and", "or", "it", "that", "this", "you", "he", "she", "we", "they"])
        else:
            logger.warning(f"Không có danh sách stopwords mặc định cho ngôn ngữ '{self.language}'.")

        if custom_stopwords:
            stopwords.update(custom_stopwords)
            logger.debug(f"Đã thêm {len(custom_stopwords)} custom stopwords.")
        return stopwords

    def _remove_stopwords(self, text: str) -> str:
        # Tách từ dựa trên khoảng trắng (giả định đã qua tokenizer)
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return " ".join(filtered_words)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self._check_input_df(X)
        cols_to_process = self._get_columns_to_process(X_transformed)

        if not self.stopwords:
            logger.warning(f"StopwordRemover không có stopwords cho '{self.language}'. Bỏ qua.")
            return X_transformed

        for col in cols_to_process:
            if col not in X_transformed.columns: continue
            series = X_transformed[col].astype(str)
            X_transformed[col] = series.apply(self._remove_stopwords)
        return X_transformed


# --- Wrapper cho TfidfVectorizer ---
class TfidfVectorizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list = None, vectorizer_params: dict = None):
        # columns: danh sách các cột text ĐÃ được tiền xử lý (clean, tokenize,...)
        # Wrapper này sẽ nối các cột này lại trước khi fit/transform TFIDF
        self.columns = columns
        self.vectorizer_params = vectorizer_params if vectorizer_params else {}
        # Đảm bảo các tham số phổ biến có giá trị mặc định nếu không được cung cấp
        self.vectorizer_params.setdefault('max_features', 5000)
        self.vectorizer_params.setdefault('ngram_range', (1, 1))


        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.fitted_columns_ = None # Lưu lại các cột đã dùng để fit
        self._feature_names_out = None # Lưu tên feature output

    def _combine_text_columns(self, X: pd.DataFrame) -> pd.Series:
        """Nối nội dung các cột text được chỉ định thành một Series duy nhất."""
        if not self.fitted_columns_:
             raise RuntimeError("Wrapper phải được fit để biết cột nào cần kết hợp.")
        # Lấy các cột thực sự có trong X tại thời điểm transform
        cols_to_combine = [col for col in self.fitted_columns_ if col in X.columns]
        if not cols_to_combine:
             raise ValueError("Không có cột nào được chỉ định trong 'columns' tồn tại trong DataFrame.")
        # Nối các cột lại, xử lý NaN bằng chuỗi rỗng
        combined_text = X[cols_to_combine].fillna('').astype(str).agg(' '.join, axis=1)
        return combined_text

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns:
            self.fitted_columns_ = [col for col in self.columns if col in X.columns]
            if not self.fitted_columns_:
                 raise ValueError("Không có cột nào trong 'columns' được chỉ định tồn tại trong DataFrame fit.")
        else:
            # Nếu không chỉ định cột, giả sử X chỉ chứa các cột text cần nối
            self.fitted_columns_ = X.columns.tolist()
            if not self.fitted_columns_:
                 raise ValueError("DataFrame đầu vào không có cột nào để fit TFIDF.")

        logger.debug(f"TfidfVectorizerWrapper fitting on columns: {self.fitted_columns_}")
        combined_text_series = self._combine_text_columns(X)
        self.vectorizer.fit(combined_text_series)
        # Lưu tên feature output sau khi fit
        self._feature_names_out = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> scipy.sparse.csr_matrix: # Trả về sparse matrix
        if self.vectorizer is None or self.fitted_columns_ is None:
            raise RuntimeError("TfidfVectorizerWrapper phải được fit trước khi transform.")

        logger.debug(f"TfidfVectorizerWrapper transforming columns: {self.fitted_columns_}")
        combined_text_series = self._combine_text_columns(X)
        transformed_data = self.vectorizer.transform(combined_text_series)
        logger.debug(f"TFIDF output shape: {transformed_data.shape}")
        return transformed_data # Trả về sparse matrix

    def get_feature_names_out(self, input_features=None):
        """Trả về tên các feature TF-IDF."""
        if self._feature_names_out is not None:
            # Có thể thêm prefix để làm rõ nguồn gốc feature
            prefix = f"tfidf_{'_'.join(self.fitted_columns_)}_" if self.fitted_columns_ else "tfidf_"
            return [f"{prefix}{name}" for name in self._feature_names_out]
        return []