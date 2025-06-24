FROM python:3.12.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
        fonts-dejavu-core \
        fontconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /workspace

# 複製並安裝 Python 依賴 (先複製 requirements.txt 利用 Docker 快取)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製應用檔案
COPY ./app ./app
COPY ./scripts ./scripts

# 下載模型
RUN python scripts/download_model.py

# 驗證模型存在
RUN ls -la models/inference_model.onnx && echo "✅ Model downloaded successfully"

# 驗證關鍵檔案存在
RUN ls -la app/_annotations.coco.json && \
    ls -la app/main.py && \
    ls -la app/config.py && \
    ls -la app/pill_detector.py && \
    echo "✅ 所有必要檔案都存在"

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]