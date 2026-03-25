# Sử dụng base image Python 3.13 phiên bản slim để tối ưu dung lượng
FROM python:3.13-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /project

# Thiết lập biến môi trường để Python không tạo ra các file .pyc và in log trực tiếp ra terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt các thư viện hệ thống cơ bản (cần thiết nếu pandas/scikit-learn yêu cầu biên dịch C)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt vào trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container (những file trong .dockerignore sẽ bị bỏ qua)
COPY . .

# Lệnh khởi chạy server FastAPI bằng Uvicorn
# Sử dụng sh -c để Docker có thể đọc được biến môi trường PORT (nếu nền tảng tự gán), mặc định là 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]