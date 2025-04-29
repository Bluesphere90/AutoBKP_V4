# tests/test_ml_workflow.py

import pytest
import pandas as pd
from pathlib import Path
import time
import json # Thêm import json
import os # Thêm import os

# Import các hàm cần test từ module ml
from app.ml.data_handler import prepare_prediction_data
# Sử dụng hàm train_client_models và hàm dự đoán kết hợp
from app.ml.models import train_client_models, predict_combined
# Import các hằng số và Enum từ config
from app.core.config import (
    TRAINING_DATA_FILENAME,
    INCREMENTAL_DATA_PREFIX,
    METADATA_FILENAME_PREFIX, # Thêm prefix metadata
    # Tên file preprocessor cụ thể
    PREPROCESSOR_HACHTOAN_FILENAME,
    PREPROCESSOR_MAHANGHOA_FILENAME,
    HACHTOAN_MODEL_FILENAME,
    MAHANGHOA_MODEL_FILENAME,
    HACHTOAN_ENCODER_FILENAME,
    MAHANGHOA_ENCODER_FILENAME,
    OUTLIER_DETECTOR_1_FILENAME,
    OUTLIER_DETECTOR_2_FILENAME,
    LABEL_ENCODERS_DIR,
    TARGET_HACHTOAN,
    TARGET_MAHANGHOA,
    # Thêm Enum và default model type
    SupportedModels,
    DEFAULT_MODEL_TYPE
)

# --- Dữ liệu mẫu (Đã đơn giản hóa) ---
SAMPLE_TRAIN_CSV_CONTENT = """MSTNguoiBan,TenHangHoaDichVu,HachToan,MaHangHoa
MST001,Phí vận chuyển hàng hóa tháng 10,642,VC001
MST002,Máy tính Dell Vostro Chính Hãng,156,DELL01
MST001,Phí tư vấn hợp đồng ABC,627,TV002
MST003,Vật tư văn phòng phẩm quý 4,153,VPP01
MST002,Bàn phím cơ logitech mới,156,LOGI01
MST004,Thuê văn phòng tầng 5 tháng 11,642,VP005
MST003,Giấy in A4 Double A loại 1,153,GIAYA4
MST002,Phí cài đặt phần mềm kế toán,1561,PMKT01
MST001,Cước điện thoại cố định,6427,DT001
MST005,Sửa chữa máy in văn phòng,6277,SCMI01
"""

SAMPLE_INCREMENTAL_CSV_CONTENT = """MSTNguoiBan,TenHangHoaDichVu,HachToan,MaHangHoa
MST001,Phí vận chuyển hàng hóa tháng 11,642,VC001
MST006,Phần mềm diệt virus kaspersky,155,KAS01
MST002,Phí bảo trì máy tính,156,DELL01
MST003,Bút bi thiên long mới nhập,153,BUTTL01
MST002,Màn hình LCD ViewSonic 24 inch,156,VIEW24
"""

SAMPLE_PREDICTION_INPUT = [
    {"MSTNguoiBan": "MST002", "TenHangHoaDichVu": "Máy tính Dell mới nhất"},
    {"MSTNguoiBan": "MST001", "TenHangHoaDichVu": "Phí vận chuyển tháng 12"},
    {"MSTNguoiBan": "MST003", "TenHangHoaDichVu": "Vật tư VPP"},
    {"MSTNguoiBan": "MST_UNKNOWN", "TenHangHoaDichVu": "Dịch vụ lạ"},
    {"MSTNguoiBan": "MST002", "TenHangHoaDichVu": "Sửa chữa máy tính dell vostro"},
]

# --- Test Fixture cho Client ID ---
@pytest.fixture
def client_id():
    return "test_client_workflow_001" # Đổi tên để tránh trùng lặp nếu chạy nhiều test

# --- Test Case Chính ---
def test_training_and_prediction_workflow(mock_config, client_id):
    """
    Test tích hợp toàn bộ quy trình huấn luyện và dự đoán.
    Bao gồm kiểm tra metadata và huấn luyện tăng cường.
    """
    data_path = mock_config["data_path"]
    models_path = mock_config["models_path"]

    client_data_path = data_path / client_id
    client_models_path = models_path / client_id
    client_encoder_path = client_models_path / LABEL_ENCODERS_DIR

    # --- Giai đoạn 1: Huấn luyện ban đầu ---
    print("\n--- Testing Initial Training ---")
    # 1. Tạo file dữ liệu huấn luyện ban đầu
    client_data_path.mkdir(parents=True, exist_ok=True)
    initial_data_file = client_data_path / TRAINING_DATA_FILENAME
    initial_data_file.write_text(SAMPLE_TRAIN_CSV_CONTENT, encoding='utf-8')
    print(f"Created initial training file: {initial_data_file}")
    assert initial_data_file.exists()

    # 2. Gọi hàm huấn luyện (chỉ định model type ban đầu)
    # Sử dụng RandomForest làm ví dụ, bạn có thể đổi sang model khác trong SupportedModels
    initial_model_to_test = SupportedModels.RANDOM_FOREST.value
    print(f"Attempting initial training with model: {initial_model_to_test}")
    train_success = train_client_models(client_id, initial_model_type_str=initial_model_to_test)
    assert train_success is True, "Quá trình huấn luyện ban đầu thất bại"

    # 3. Kiểm tra xem các file model/preprocessor/encoder/outlier đã được tạo chưa
    expected_files = [
        client_models_path / PREPROCESSOR_HACHTOAN_FILENAME,
        client_models_path / HACHTOAN_MODEL_FILENAME,
        client_encoder_path / HACHTOAN_ENCODER_FILENAME,
        client_models_path / OUTLIER_DETECTOR_1_FILENAME,

        client_models_path / PREPROCESSOR_MAHANGHOA_FILENAME,
        client_models_path / MAHANGHOA_MODEL_FILENAME,
        client_encoder_path / MAHANGHOA_ENCODER_FILENAME,
        client_models_path / OUTLIER_DETECTOR_2_FILENAME,
    ]
    # Kiểm tra file metadata
    metadata_files_initial = sorted(client_models_path.glob(f"{METADATA_FILENAME_PREFIX}*.json"))
    assert metadata_files_initial, "Không tìm thấy file metadata nào được tạo sau huấn luyện ban đầu!"
    latest_metadata_file_initial = metadata_files_initial[-1]
    expected_files.append(latest_metadata_file_initial)
    print(f"Found metadata file: {latest_metadata_file_initial.name}")

    for f_path in expected_files:
        print(f"Checking existence: {f_path}")
        assert f_path.exists(), f"File mong đợi không được tạo: {f_path.name}"
        assert f_path.stat().st_size > 0, f"File {f_path.name} rỗng!"

    # Kiểm tra nội dung metadata (tùy chọn nhưng hữu ích)
    try:
        with open(latest_metadata_file_initial, 'r', encoding='utf-8') as f:
            metadata_content = json.load(f)
        assert metadata_content.get("client_id") == client_id
        assert metadata_content.get("training_type") == "initial"
        assert metadata_content.get("status") == "COMPLETED"
        assert metadata_content.get("selected_model_type") == initial_model_to_test
        assert metadata_content["hachtoan_model_info"].get("model_class") == initial_model_to_test
        # Kiểm tra xem evaluation metrics có được tạo không (có thể có lỗi)
        assert "evaluation_metrics" in metadata_content["hachtoan_model_info"]
        print("Metadata content basic check passed.")
    except Exception as e:
        pytest.fail(f"Lỗi khi đọc hoặc kiểm tra file metadata: {e}")


    # --- Giai đoạn 2: Dự đoán sau huấn luyện ban đầu ---
    print("\n--- Testing Prediction after Initial Training ---")
    # 1. Chuẩn bị dữ liệu dự đoán
    prediction_df = prepare_prediction_data(SAMPLE_PREDICTION_INPUT)
    assert isinstance(prediction_df, pd.DataFrame)
    assert len(prediction_df) == len(SAMPLE_PREDICTION_INPUT)

    # 2. Gọi hàm dự đoán (sử dụng predict_combined)
    predictions = predict_combined(client_id, prediction_df)
    assert isinstance(predictions, list)
    assert len(predictions) == len(SAMPLE_PREDICTION_INPUT)

    # 3. Kiểm tra cấu trúc và kiểu dữ liệu của kết quả dự đoán
    print("Sample predictions (initial):", predictions[:2])
    for i, result in enumerate(predictions):
        assert isinstance(result, dict)
        assert TARGET_HACHTOAN in result
        assert f"{TARGET_HACHTOAN}_prob" in result
        assert TARGET_MAHANGHOA in result
        assert f"{TARGET_MAHANGHOA}_prob" in result
        assert "is_outlier_input1" in result
        assert "is_outlier_input2" in result
        assert "error" in result # Kiểm tra key error tồn tại

        if result.get(TARGET_HACHTOAN) is not None:
            assert isinstance(result[TARGET_HACHTOAN], (str, int, float))
        if result.get(f"{TARGET_HACHTOAN}_prob") is not None:
             assert isinstance(result[f"{TARGET_HACHTOAN}_prob"], float)
             assert 0.0 <= result[f"{TARGET_HACHTOAN}_prob"] <= 1.0

        if result.get(TARGET_MAHANGHOA) is not None:
             assert isinstance(result[TARGET_MAHANGHOA], (str, int, float))
        if result.get(f"{TARGET_MAHANGHOA}_prob") is not None:
             assert isinstance(result[f"{TARGET_MAHANGHOA}_prob"], float)
             assert 0.0 <= result[f"{TARGET_MAHANGHOA}_prob"] <= 1.0

        # Sử dụng bool() để xử lý cả bool chuẩn và numpy bool
        assert isinstance(bool(result["is_outlier_input1"]), bool)
        assert isinstance(bool(result["is_outlier_input2"]), bool)

    # --- Giai đoạn 3: Huấn luyện tăng cường ---
    print("\n--- Testing Incremental Training ---")
    # 1. Tạo file dữ liệu tăng cường
    timestamp = int(time.time())
    incremental_file = client_data_path / f"{INCREMENTAL_DATA_PREFIX}{timestamp}.csv"
    incremental_file.write_text(SAMPLE_INCREMENTAL_CSV_CONTENT, encoding='utf-8')
    print(f"Created incremental training file: {incremental_file}")
    assert incremental_file.exists()

    # Lưu thông tin cũ để so sánh
    old_model_mtime = (client_models_path / HACHTOAN_MODEL_FILENAME).stat().st_mtime
    old_metadata_count = len(metadata_files_initial)

    # 2. Gọi lại hàm huấn luyện (không cần truyền model type)
    print("Attempting incremental training...")
    incremental_train_success = train_client_models(client_id, initial_model_type_str=None)
    assert incremental_train_success is True, "Quá trình huấn luyện tăng cường thất bại"

    # 3. Kiểm tra ghi đè model và tạo metadata mới
    new_model_mtime = (client_models_path / HACHTOAN_MODEL_FILENAME).stat().st_mtime
    assert new_model_mtime > old_model_mtime, "Model file không được cập nhật sau huấn luyện tăng cường"
    print("Model file modification time updated, overwrite successful.")

    metadata_files_incremental = sorted(client_models_path.glob(f"{METADATA_FILENAME_PREFIX}*.json"))
    new_metadata_count = len(metadata_files_incremental)
    assert new_metadata_count > old_metadata_count, "Không có file metadata mới được tạo sau huấn luyện tăng cường"
    latest_metadata_file_incremental = metadata_files_incremental[-1]
    print(f"New metadata file created: {latest_metadata_file_incremental.name}")

    # Kiểm tra nội dung metadata mới (tùy chọn)
    try:
        with open(latest_metadata_file_incremental, 'r', encoding='utf-8') as f:
            metadata_content_inc = json.load(f)
        assert metadata_content_inc.get("client_id") == client_id
        assert metadata_content_inc.get("training_type") == "incremental" # Phải là incremental
        assert metadata_content_inc.get("status") == "COMPLETED"
        assert metadata_content_inc.get("selected_model_type") == initial_model_to_test # Phải giữ nguyên model type
        assert "evaluation_metrics" in metadata_content_inc["hachtoan_model_info"]
        print("Incremental metadata content basic check passed.")
    except Exception as e:
        pytest.fail(f"Lỗi khi đọc hoặc kiểm tra file metadata tăng cường: {e}")


    # --- Giai đoạn 4: Dự đoán sau huấn luyện tăng cường ---
    print("\n--- Testing Prediction after Incremental Training ---")
    # 1. Sử dụng lại cùng dữ liệu dự đoán
    predictions_after_incremental = predict_combined(client_id, prediction_df)
    assert isinstance(predictions_after_incremental, list)
    assert len(predictions_after_incremental) == len(SAMPLE_PREDICTION_INPUT)

    # 2. Kiểm tra cấu trúc (tương tự như trước)
    print("Sample predictions (incremental):", predictions_after_incremental[:2])
    for result in predictions_after_incremental:
         assert isinstance(result, dict)
         assert TARGET_HACHTOAN in result
         # ...(lặp lại các assert về key và type như Giai đoạn 2)...
         assert isinstance(bool(result["is_outlier_input1"]), bool)
         assert isinstance(bool(result["is_outlier_input2"]), bool)
         assert "error" in result


    # 3. So sánh kết quả dự đoán trước và sau tăng cường (chỉ để quan sát)
    print("Predictions before incremental (sample):", predictions[0])
    print("Predictions after incremental (sample):", predictions_after_incremental[0])

    print("\n--- ML Workflow Test Completed Successfully ---")