# 多階段構建 - 輕量化推理環境
FROM python:3.12.11-slim AS builder

# 安裝構建依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gcc \
        g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# 生產階段 - 最小化映像檔
FROM python:3.12.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 只安裝運行時必需的系統依賴 (已移除 OpenCV 相關依賴)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        fonts-dejavu-core \
        fontconfig \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /workspace

# 從構建階段複製 wheels，安裝依賴，下載模型，設置用戶 - 一次完成
COPY --from=builder /build/wheels /wheels
COPY requirements.txt /wheels/requirements.txt
COPY main.py uvicorn.prod.py ./
COPY ./app ./app
COPY ./scripts/download_model.py /tmp/

RUN pip install --no-cache-dir --find-links /wheels -r /wheels/requirements.txt && \
    rm -rf /wheels /root/.cache/pip && \
    python /tmp/download_model.py && \
    rm -f /tmp/download_model.py && \
    ls -la models/inference_model.onnx && \
    groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8000

# 輕量化健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# 優化的 ASGI 啟動
CMD ["python", "uvicorn.prod.py"]