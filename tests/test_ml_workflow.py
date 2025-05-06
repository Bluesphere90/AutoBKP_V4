# tests/test_ml_workflow.py

import pytest
import pandas as pd
from pathlib import Path
import time
import json
import os

# Import các hàm cần test từ module ml
from app.ml.data_handler import prepare_prediction_data
# Sử dụng hàm train và các hàm predict đã cập nhật
from app.ml.models import (
    train_client_models,
    predict_combined,
    predict_hachtoan_only,
    predict_mahanghoa_only,
    predict_mahanghoa_direct # Hàm mới
)
# Import các hằng số và Enum từ config
from app.core.config import (
    TRAINING_DATA_FILENAME, INCREMENTAL_DATA_PREFIX, METADATA_FILENAME_PREFIX,
    PREPROCESSOR_HACHTOAN_FILENAME, PREPROCESSOR_MAHANGHOA_FILENAME, # Tên prep cụ thể
    HACHTOAN_MODEL_FILENAME, MAHANGHOA_MODEL_FILENAME, MAHANGHOA_DIRECT_MODEL_FILENAME, # Thêm model mới
    HACHTOAN_ENCODER_FILENAME, MAHANGHOA_ENCODER_FILENAME,
    OUTLIER_DETECTOR_1_FILENAME, OUTLIER_DETECTOR_2_FILENAME,
    LABEL_ENCODERS_DIR, TARGET_HACHTOAN, TARGET_MAHANGHOA,
    SupportedModels, DEFAULT_MODEL_TYPE,
    SupportedColumnType, SupportedLanguage, CategoricalStrategy, TextVectorizerStrategy
)
# Import config để dùng tên file config cột
import app.core.config as core_config

# --- Dữ liệu mẫu ---
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
SAMPLE_MH_DEPENDENT_INPUT = [
     {"MSTNguoiBan": "MST002", "TenHangHoaDichVu": "Bàn phím cơ logitech mới", "HachToan": "156"},
     {"MSTNguoiBan": "MST002", "TenHangHoaDichVu": "Phí cài đặt", "HachToan": "1561"},
]

# --- Metadata Cột Mẫu ---
SAMPLE_COLUMN_CONFIG = {
  "columns": {
    "MSTNguoiBan": {"type": SupportedColumnType.CATEGORICAL.value, "strategy": CategoricalStrategy.ONEHOT.value, "handle_unknown": "ignore"},
    "TenHangHoaDichVu": {"type": SupportedColumnType.TEXT.value, "language": SupportedLanguage.VIETNAMESE.value, "vectorizer": TextVectorizerStrategy.TFIDF.value, "max_features": 5000, "ngram_range": (1, 2)},
    "HachToan": {"type": SupportedColumnType.CATEGORICAL.value, "strategy": CategoricalStrategy.ONEHOT.value, "handle_unknown": "ignore"},
  },
  "remainder_strategy": "drop"
}

# --- Test Fixture cho Client ID ---
@pytest.fixture
def client_id():
    # Sử dụng tên mới để đảm bảo không bị ảnh hưởng bởi lần chạy trước nếu thư mục temp không được dọn dẹp hoàn toàn
    return f"test_client_full_workflow_{int(time.time())}"

# --- Test Case Chính (Cập nhật) ---
def test_training_and_prediction_workflow(mock_config, client_id):
    """Test tích hợp huấn luyện 3 models và các kịch bản dự đoán."""
    data_path = mock_config["data_path"]
    models_path = mock_config["models_path"]

    client_data_path = data_path / client_id
    client_models_path = models_path / client_id
    client_encoder_path = client_models_path / LABEL_ENCODERS_DIR
    # Tạo thư mục model client trước (mkdir trong utils đã làm việc này, nhưng gọi lại cho chắc)
    client_models_path.mkdir(parents=True, exist_ok=True)
    client_encoder_path.mkdir(parents=True, exist_ok=True) # Đảm bảo cả thư mục encoder

    # --- Giai đoạn 0: Tạo file cấu hình cột ---
    print("\n--- Creating Column Config File ---")
    column_config_file = client_models_path / "column_config.json"
    # Đảm bảo thư mục models tồn tại trước khi ghi config
    column_config_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(column_config_file, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_COLUMN_CONFIG, f, ensure_ascii=False, indent=4)
        print(f"Created column config file: {column_config_file}")
        assert column_config_file.exists(), "Không tạo được file column_config.json"
    except Exception as e:
        pytest.fail(f"Không thể tạo file column_config.json mẫu: {e}")

    # --- Giai đoạn 1: Huấn luyện ban đầu ---
    print("\n--- Testing Initial Training ---")
    # 1. Tạo file dữ liệu huấn luyện ban đầu
    initial_data_file = client_data_path / TRAINING_DATA_FILENAME
    # Đảm bảo thư mục data tồn tại NGAY TRƯỚC KHI GHI
    print(f"Ensuring data directory exists: {initial_data_file.parent}")
    initial_data_file.parent.mkdir(parents=True, exist_ok=True)
    initial_data_file.write_text(SAMPLE_TRAIN_CSV_CONTENT, encoding='utf-8')
    print(f"Created initial training file: {initial_data_file}")
    assert initial_data_file.exists(), "Không tạo được file training_data.csv"

    # 2. Gọi hàm huấn luyện (chỉ định model type ban đầu)
    initial_model_to_test = SupportedModels.RANDOM_FOREST.value
    print(f"Attempting initial training with model: {initial_model_to_test}")
    train_success = train_client_models(client_id, initial_model_type_str=initial_model_to_test)
    assert train_success is True, f"Hàm train_client_models trả về False cho huấn luyện ban đầu."

    # 3. Kiểm tra xem các file đã được tạo chưa
    expected_files = [
        client_models_path / PREPROCESSOR_HACHTOAN_FILENAME,
        client_models_path / PREPROCESSOR_MAHANGHOA_FILENAME,
        client_models_path / HACHTOAN_MODEL_FILENAME,
        client_models_path / MAHANGHOA_MODEL_FILENAME,
        client_models_path / MAHANGHOA_DIRECT_MODEL_FILENAME,
        client_encoder_path / HACHTOAN_ENCODER_FILENAME,
        client_encoder_path / MAHANGHOA_ENCODER_FILENAME,
        client_models_path / OUTLIER_DETECTOR_1_FILENAME,
        client_models_path / OUTLIER_DETECTOR_2_FILENAME,
    ]
    metadata_files_initial = sorted(client_models_path.glob(f"{METADATA_FILENAME_PREFIX}*.json"))
    assert metadata_files_initial, "Không tìm thấy file metadata nào được tạo sau huấn luyện ban đầu!"
    latest_metadata_file_initial = metadata_files_initial[-1]
    expected_files.append(latest_metadata_file_initial)
    print(f"Found metadata file: {latest_metadata_file_initial.name}")

    missing_files = []
    empty_files = []
    for f_path in expected_files:
        print(f"Checking existence and size: {f_path}")
        if not f_path.exists():
            missing_files.append(f_path.name)
        elif f_path.stat().st_size == 0:
            empty_files.append(f_path.name)

    assert not missing_files, f"Các file sau không được tạo: {', '.join(missing_files)}"
    assert not empty_files, f"Các file sau bị rỗng: {', '.join(empty_files)}"

    # Kiểm tra nội dung metadata ban đầu
    try:
        with open(latest_metadata_file_initial, 'r', encoding='utf-8') as f: metadata_content = json.load(f)
        assert metadata_content.get("client_id") == client_id
        assert metadata_content.get("training_type") == "initial"
        assert metadata_content.get("status") == "COMPLETED"
        assert metadata_content.get("selected_model_type") == initial_model_to_test
        assert metadata_content.get("column_config_file") == "column_config.json"
        assert "mahanghoa_direct_model_info" in metadata_content
        assert metadata_content["mahanghoa_direct_model_info"].get("model_saved") is True
        assert metadata_content["hachtoan_model_info"].get("preprocessor_file") == PREPROCESSOR_HACHTOAN_FILENAME
        assert "evaluation_metrics" in metadata_content["hachtoan_model_info"]
        print("Initial metadata content basic check passed.")
    except Exception as e:
        pytest.fail(f"Lỗi khi đọc hoặc kiểm tra file metadata ban đầu: {e}")


    # --- Giai đoạn 2: Dự đoán sau huấn luyện ban đầu ---
    print("\n--- Testing Predictions after Initial Training ---")
    # 2.1 Test predict_combined
    prediction_df_base = prepare_prediction_data(SAMPLE_PREDICTION_INPUT)
    assert isinstance(prediction_df_base, pd.DataFrame) and not prediction_df_base.empty, "Prepare prediction data failed"
    print(f"Predicting combined for {len(prediction_df_base)} items.")
    predictions_combined = predict_combined(client_id, prediction_df_base.copy())
    assert isinstance(predictions_combined, list), "predict_combined did not return a list"
    assert len(predictions_combined) == len(SAMPLE_PREDICTION_INPUT), f"Expected {len(SAMPLE_PREDICTION_INPUT)} combined predictions, got {len(predictions_combined)}"
    print("Sample combined predictions:", predictions_combined[:1])
    # ... (Thêm assert chi tiết nếu cần)

    # 2.2 Test predict_hachtoan_only
    print(f"Predicting hachtoan_only for {len(prediction_df_base)} items.")
    predictions_ht = predict_hachtoan_only(client_id, prediction_df_base.copy())
    assert isinstance(predictions_ht, list), "predict_hachtoan_only did not return a list"
    assert len(predictions_ht) == len(SAMPLE_PREDICTION_INPUT), f"Expected {len(SAMPLE_PREDICTION_INPUT)} hachtoan_only predictions, got {len(predictions_ht)}"
    print("Sample hachtoan_only predictions:", predictions_ht[:1])
    for r in predictions_ht: assert TARGET_MAHANGHOA not in r

    # 2.3 Test predict_mahanghoa_only (dependent)
    prediction_df_mh_dep = prepare_prediction_data(SAMPLE_MH_DEPENDENT_INPUT)
    assert isinstance(prediction_df_mh_dep, pd.DataFrame) and not prediction_df_mh_dep.empty, "Prepare MH dependent prediction data failed"
    print(f"Predicting mahanghoa_only (dependent) for {len(prediction_df_mh_dep)} items.")
    predictions_mh_dep = predict_mahanghoa_only(client_id, prediction_df_mh_dep)
    assert isinstance(predictions_mh_dep, list), "predict_mahanghoa_only did not return a list"
    assert len(predictions_mh_dep) == len(SAMPLE_MH_DEPENDENT_INPUT), f"Expected {len(SAMPLE_MH_DEPENDENT_INPUT)} MH dependent predictions, got {len(predictions_mh_dep)}"
    print("Sample mahanghoa_only (dependent) predictions:", predictions_mh_dep[:1])
    for r in predictions_mh_dep: assert TARGET_HACHTOAN not in r

    # 2.4 Test predict_mahanghoa_direct (MỚI)
    print(f"Predicting mahanghoa_direct for {len(prediction_df_base)} items.")
    predictions_mh_direct = predict_mahanghoa_direct(client_id, prediction_df_base.copy())
    assert isinstance(predictions_mh_direct, list), "predict_mahanghoa_direct did not return a list"
    assert len(predictions_mh_direct) == len(SAMPLE_PREDICTION_INPUT), f"Expected {len(SAMPLE_PREDICTION_INPUT)} MH direct predictions, got {len(predictions_mh_direct)}"
    print("Sample mahanghoa_direct predictions:", predictions_mh_direct[:1])
    for r in predictions_mh_direct:
        assert TARGET_HACHTOAN not in r
        assert TARGET_MAHANGHOA in r
        assert "is_outlier_input1" in r
        assert "is_outlier_input2" not in r


    # --- Giai đoạn 3: Huấn luyện tăng cường ---
    print("\n--- Testing Incremental Training ---")
    incremental_file = client_data_path / f"{INCREMENTAL_DATA_PREFIX}{int(time.time())}.csv"
    incremental_file.parent.mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục data tồn tại
    incremental_file.write_text(SAMPLE_INCREMENTAL_CSV_CONTENT, encoding='utf-8')
    print(f"Created incremental training file: {incremental_file}")
    assert incremental_file.exists()

    # Lưu thông tin cũ
    old_ht_model_mtime = (client_models_path / HACHTOAN_MODEL_FILENAME).stat().st_mtime
    old_mh_model_mtime = (client_models_path / MAHANGHOA_MODEL_FILENAME).stat().st_mtime
    old_mh_direct_model_mtime = (client_models_path / MAHANGHOA_DIRECT_MODEL_FILENAME).stat().st_mtime
    old_prep_ht_mtime = (client_models_path / PREPROCESSOR_HACHTOAN_FILENAME).stat().st_mtime
    old_prep_mh_mtime = (client_models_path / PREPROCESSOR_MAHANGHOA_FILENAME).stat().st_mtime
    old_metadata_count = len(metadata_files_initial)

    # Gọi lại hàm huấn luyện
    print("Attempting incremental training...")
    incremental_train_success = train_client_models(client_id, initial_model_type_str=None)
    assert incremental_train_success is True, "Huấn luyện tăng cường thất bại"

    # Kiểm tra cập nhật models và metadata, KHÔNG cập nhật preprocessors
    new_ht_model_mtime = (client_models_path / HACHTOAN_MODEL_FILENAME).stat().st_mtime
    new_mh_model_mtime = (client_models_path / MAHANGHOA_MODEL_FILENAME).stat().st_mtime
    new_mh_direct_model_mtime = (client_models_path / MAHANGHOA_DIRECT_MODEL_FILENAME).stat().st_mtime
    assert new_ht_model_mtime > old_ht_model_mtime, "Model HT không được cập nhật"
    assert new_mh_model_mtime > old_mh_model_mtime, "Model MH không được cập nhật"
    assert new_mh_direct_model_mtime > old_mh_direct_model_mtime, "Model MH Direct không được cập nhật"
    print("Model files modification times updated.")

    new_prep_ht_mtime = (client_models_path / PREPROCESSOR_HACHTOAN_FILENAME).stat().st_mtime
    new_prep_mh_mtime = (client_models_path / PREPROCESSOR_MAHANGHOA_FILENAME).stat().st_mtime
    assert new_prep_ht_mtime == old_prep_ht_mtime, "Preprocessor HT bị ghi đè khi huấn luyện tăng cường!"
    assert new_prep_mh_mtime == old_prep_mh_mtime, "Preprocessor MH bị ghi đè khi huấn luyện tăng cường!"
    print("Preprocessor files modification times unchanged (correct).")

    metadata_files_incremental = sorted(client_models_path.glob(f"{METADATA_FILENAME_PREFIX}*.json"))
    new_metadata_count = len(metadata_files_incremental)
    assert new_metadata_count > old_metadata_count, "Metadata mới không được tạo"
    latest_metadata_file_incremental = metadata_files_incremental[-1]
    print(f"New metadata file created: {latest_metadata_file_incremental.name}")

    # Kiểm tra nội dung metadata tăng cường
    try:
        with open(latest_metadata_file_incremental, 'r', encoding='utf-8') as f: metadata_content_inc = json.load(f)
        assert metadata_content_inc.get("training_type") == "incremental"
        assert metadata_content_inc.get("selected_model_type") == initial_model_to_test
        print("Incremental metadata basic check passed.")
    except Exception as e: pytest.fail(f"Lỗi kiểm tra metadata tăng cường: {e}")


    # --- Giai đoạn 4: Dự đoán sau huấn luyện tăng cường ---
    print("\n--- Testing Prediction after Incremental Training ---")
    print(f"Predicting combined for {len(prediction_df_base)} items after incremental.")
    predictions_after_inc = predict_combined(client_id, prediction_df_base.copy())
    assert isinstance(predictions_after_inc, list), "predict_combined after incremental failed"
    assert len(predictions_after_inc) == len(SAMPLE_PREDICTION_INPUT), "Incorrect number of predictions after incremental"
    print("Sample combined predictions (incremental):", predictions_after_inc[:1])
    # ...(Assert chi tiết nếu cần)...

    print("\n--- ML Workflow Test Completed Successfully ---")