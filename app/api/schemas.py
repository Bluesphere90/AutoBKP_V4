# app/api/schemas.py

from pydantic import BaseModel, Field, validator # Sử dụng validator cho Pydantic V1
from typing import List, Dict, Any, Optional, Union, Tuple

# Import Enums từ config
from app.core.config import SupportedColumnType, SupportedLanguage, CategoricalStrategy, NumericalStrategy, TextVectorizerStrategy

# --- Schemas Input gốc ---
class PredictionInputItem(BaseModel):
    MSTNguoiBan: str = Field(..., example="MST001")
    TenHangHoaDichVu: str = Field(..., example="Phí vận chuyển")
    additional_data: Dict[str, Any] = Field({}, description="Dữ liệu bổ sung")
    def to_flat_dict(self) -> Dict[str, Any]:
        data = self.dict(exclude={'additional_data'}) # Pydantic V1
        data.update(self.additional_data)
        return data
class PredictionRequest(BaseModel):
    items: List[PredictionInputItem] = Field(..., min_items=1)

# --- Schema Input cho dự đoán chỉ MaHangHoa (Dependent) ---
class MaHangHoaPredictionInputItem(PredictionInputItem):
    HachToan: str = Field(..., example="156")
class MaHangHoaPredictionRequest(BaseModel):
    items: List[MaHangHoaPredictionInputItem] = Field(..., min_items=1)


# --- Schemas Output gốc (Kết hợp) ---
class PredictionResultItem(BaseModel):
    HachToan: Optional[str] = Field(None, example="642")
    HachToan_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    MaHangHoa: Optional[str] = Field(None, example="VC001")
    MaHangHoa_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_outlier_input1: bool = Field(...)
    is_outlier_input2: bool = Field(...) # Outlier cho input MH Dependent
    error: Optional[str] = Field(None)
class PredictionResponse(BaseModel):
    results: List[PredictionResultItem]

# --- Schema Output chỉ cho HachToan ---
class HachToanPredictionResultItem(BaseModel):
    HachToan: Optional[str] = Field(None, example="642")
    HachToan_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_outlier_input1: bool = Field(...)
    error: Optional[str] = Field(None)
class HachToanPredictionResponse(BaseModel):
    results: List[HachToanPredictionResultItem]

# --- Schema Output chỉ cho MaHangHoa (Dependent) ---
class MaHangHoaPredictionResultItem(BaseModel):
    MaHangHoa: Optional[str] = Field(None, example="DELL01")
    MaHangHoa_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_outlier_input2: bool = Field(...) # Outlier cho input MH Dependent
    error: Optional[str] = Field(None)
class MaHangHoaPredictionResponse(BaseModel):
    results: List[MaHangHoaPredictionResultItem]

# --- Schema Output MỚI chỉ cho MaHangHoa (Direct) ---
class MaHangHoaDirectPredictionResultItem(BaseModel):
    """Schema kết quả khi dự đoán MaHangHoa trực tiếp."""
    MaHangHoa: Optional[str] = Field(None, example="VPP01", description="Kết quả dự đoán MaHangHoa (trực tiếp)")
    MaHangHoa_prob: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.65, description="Xác suất của MaHangHoa dự đoán (trực tiếp)")
    # Sử dụng outlier flag của input gốc (input1) vì dùng preprocessor_ht
    is_outlier_input1: bool = Field(..., example=False, description="True nếu input gốc bị coi là outlier")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có cho dòng này")

class MaHangHoaDirectPredictionResponse(BaseModel):
    """Schema cho response chỉ dự đoán MaHangHoa (trực tiếp)."""
    results: List[MaHangHoaDirectPredictionResultItem] = Field(..., description="Danh sách kết quả dự đoán MaHangHoa (trực tiếp)")


# --- Schemas cho Column Configuration (Dùng validator V1) ---
class ColumnConfigItem(BaseModel):
    type: SupportedColumnType
    language: Optional[SupportedLanguage] = None
    strategy: Optional[Union[CategoricalStrategy, NumericalStrategy, TextVectorizerStrategy, str]] = None
    handle_unknown: Optional[str] = None
    imputer_strategy: Optional[str] = None
    scaler: Optional[str] = None
    vectorizer: Optional[str] = None
    max_features: Optional[int] = None
    ngram_range: Optional[Tuple[int, int]] = None
    n_features: Optional[int] = None

    @validator('language')
    def language_usage(cls, v, values):
        col_type = values.get('type')
        if col_type == SupportedColumnType.TEXT and v is None: return SupportedLanguage.NONE
        if col_type != SupportedColumnType.TEXT and v is not None: raise ValueError("'language' chỉ dùng cho type='text'")
        return v
    # Thêm các validators khác nếu cần

class ColumnConfigRequest(BaseModel):
    columns: Dict[str, ColumnConfigItem]
    default_text_language: Optional[SupportedLanguage] = None
    remainder_strategy: Optional[str] = Field("passthrough")

    @validator('columns')
    def check_required_columns(cls, v):
        from app.core.config import INPUT_COLUMNS
        defined_cols = v.keys()
        missing_required = [col for col in INPUT_COLUMNS if col not in defined_cols]
        if missing_required: raise ValueError(f"Thiếu cột input bắt buộc: {missing_required}")
        for col in INPUT_COLUMNS:
            if col in v and isinstance(v[col], ColumnConfigItem) and v[col].type == SupportedColumnType.IGNORE:
                 raise ValueError(f"Cột input bắt buộc '{col}' không được là 'ignore'.")
        return v
ColumnConfigResponse = ColumnConfigRequest


# --- Schemas cho Thông báo Chung ---
class MessageResponse(BaseModel):
    message: str
    client_id: Optional[str] = None
    status_code: Optional[int] = 200
class ErrorDetail(BaseModel):
    loc: Optional[List[Union[str, int]]] = None
    msg: str
    type: Optional[str] = None
class ErrorResponse(BaseModel):
    detail: Union[str, List[Dict[str, Any]]] # Pydantic V1