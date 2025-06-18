FROM python:3.12.11-slim

# 設置環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 安裝系統依賴和字體
RUN apt-get update && apt-get install -y \
    # OpenCV 依賴
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    # 網路工具 (用於健康檢查)
    curl \
    # 黑體字形包
    fonts-dejavu \
    fonts-dejavu-extra \
    fonts-liberation \
    fonts-noto \
    fontconfig \
    # 清理快取
    && fc-cache -fv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製並安裝 Python 依賴 (先複製 requirements.txt 利用 Docker 快取)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製應用檔案
COPY ./app ./app
COPY ./models ./models

# 驗證關鍵檔案存在
RUN ls -la models/inference_model.onnx && \
    ls -la app/_annotations.coco.json && \
    ls -la app/main.py && \
    echo "✅ 所有必要檔案都存在"

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]