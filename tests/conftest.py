import pytest
import shutil
from pathlib import Path

@pytest.fixture(scope="function") # scope="function": chạy lại fixture cho mỗi test function
def temp_data_dir(tmp_path_factory):
    """Fixture để tạo thư mục dữ liệu tạm thời."""
    # tmp_path_factory là fixture sẵn có của pytest để tạo thư mục tạm duy nhất
    temp_dir = tmp_path_factory.mktemp("client_data_test")
    print(f"Created temp data dir: {temp_dir}")
    yield temp_dir # Trả về đường dẫn thư mục tạm
    # Cleanup (thường không cần vì pytest tự dọn dẹp tmp_path_factory)
    # print(f"Removing temp data dir: {temp_dir}")
    # shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def temp_models_dir(tmp_path_factory):
    """Fixture để tạo thư mục models tạm thời."""
    temp_dir = tmp_path_factory.mktemp("client_models_test")
    print(f"Created temp models dir: {temp_dir}")
    yield temp_dir
    # Cleanup (thường không cần)
    # print(f"Removing temp models dir: {temp_dir}")
    # shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def mock_config(monkeypatch, temp_data_dir, temp_models_dir):
    """Fixture để patch đường dẫn trong config sử dụng thư mục tạm."""
    # monkeypatch là fixture của pytest để thay đổi giá trị của module/class/function
    # trong suốt quá trình test
    import app.core.config as config

    print(f"Patching config: BASE_DATA_PATH -> {temp_data_dir}")
    monkeypatch.setattr(config, "BASE_DATA_PATH", temp_data_dir)

    print(f"Patching config: BASE_MODELS_PATH -> {temp_models_dir}")
    monkeypatch.setattr(config, "BASE_MODELS_PATH", temp_models_dir)

    # Trả về các đường dẫn đã patch để có thể sử dụng trong test nếu cần
    return {"data_path": temp_data_dir, "models_path": temp_models_dir}