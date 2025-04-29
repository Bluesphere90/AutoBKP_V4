# Dự án Dự đoán Hạch Toán & Mã Hàng Hóa

Dự án này sử dụng Machine Learning để dự đoán các tài khoản kế toán (`HachToan`) và mã hàng hóa (`MaHangHoa`) dựa trên Mã số thuế người bán (`MSTNguoiBan`) và Tên hàng hóa/dịch vụ (`TenHangHoaDichVu`).

## Tính năng chính

*   Phân loại đa lớp (Multi-class classification).
*   Xử lý văn bản tiếng Việt.
*   Huấn luyện mô hình tuần tự: Dự đoán `HachToan` trước, sau đó sử dụng kết quả để dự đoán `MaHangHoa` (nếu `HachToan` bắt đầu bằng "15").
*   Hỗ trợ huấn luyện ban đầu và huấn luyện tăng cường (retrain) hàng tháng.
*   Phát hiện điểm bất thường (outlier) trong dữ liệu dự đoán và cảnh báo.
*   Cung cấp API để huấn luyện và dự đoán cho từng khách hàng.
*   Triển khai bằng Docker Compose với lưu trữ dữ liệu và mô hình bền vững qua Docker Volumes.

