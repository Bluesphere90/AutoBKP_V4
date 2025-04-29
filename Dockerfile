# Dockerfile

# Chọn một base image Python chính thức. Sử dụng phiên bản cụ thể là tốt nhất.
# Đảm bảo phiên bản Python này tương thích với code và thư viện của bạn (ví dụ: 3.10, 3.11)
# Sử dụng slim-buster để image nhỏ hơn
FROM python:3.11-slim-buster

# Đặt biến môi trường để Python không buffer stdout/stderr, giúp log hiển thị ngay lập tức
ENV PYTHONUNBUFFERED 1
# Đặt biến môi trường cho đường dẫn data/models bên trong container (có thể ghi đè bằng docker-compose)
ENV DATA_PATH=/app/data/client_data
ENV MODELS_PATH=/app/models/client_models

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Cập nhật pip và cài đặt các gói cần thiết
# Copy file requirements TRƯỚC khi copy code để tận dụng Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code từ thư mục hiện tại vào thư mục /app trong container
COPY ./app /app/app

# (Nếu bạn có file .env và muốn copy vào image - không khuyến nghị cho secrets)
# COPY .env .

# Expose cổng mà FastAPI/uvicorn sẽ chạy (ví dụ: 8000)
EXPOSE 8000

# Lệnh để chạy ứng dụng khi container khởi động
# Sử dụng uvicorn để chạy ASGI app (app.api.main:app)
# --host 0.0.0.0: Cho phép truy cập từ bên ngoài container
# --port 8000: Cổng lắng nghe bên trong container
# --reload: Tự động reload khi code thay đổi (CHỈ DÙNG CHO DEVELOPMENT, xóa hoặc comment dòng này cho PRODUCTION)
# CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Lệnh chạy cho PRODUCTION (không có --reload)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]