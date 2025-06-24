# 💊 藥丸檢測 API

使用 RF-DETR ONNX 模型進行藥品識別的 FastAPI 應用程式，支援網頁介面和 RESTful API。

## 🚀 線上體驗

直接打開網頁，選擇藥物圖片上傳即可辨識：

👉 [https://pill-detector-23010935669.us-central1.run.app/test](https://pill-detector-23010935669.us-central1.run.app/test)

## 🏥 支援的藥物類別

目前支援 6 種藥物：
- **Amoxicillin** 
- **Diovan 160mg** 
- **Lansoprazole** 
- **Relecox** 
- **Takepron** 
- **Utraphen** 

## 📁 專案架構

```
app/
├── main.py                # 主應用程式
├── pill_detector.py       # 檢測核心邏輯
├── config.py              # 統一配置管理
├── main_legacy.py         # 原始版本
├── test.py                # API 測試腳本
└── _annotations.coco.json # COCO 格式類別定義
```

## 🐳 Docker 使用方式

### 方法一：使用預建映像檔

1. **拉取映像檔**：
   ```bash
   docker pull ghcr.io/wenalyzer/pill-detector:latest
   ```

2. **啟動服務**：
   ```bash
   docker run -d -p 8000:8000 ghcr.io/wenalyzer/pill-detector:latest
   ```

3. **開啟網頁介面**：
   ```
   http://localhost:8000/test
   ```

### 方法二：本地建置

1. **建置映像檔**：
   ```bash
   docker build -t pill-detector .
   ```

2. **執行容器**：
   ```bash
   docker run -d -p 8000:8000 pill-detector
   ```

### 啟動應用程式

```bash
# 使用新版（推薦）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 或使用原始完整版
uvicorn app.main_legacy:app --host 0.0.0.0 --port 8000 --reload
```

## 🔗 API 端點

- `GET /` - API 狀態資訊
- `GET /health` - 健康檢查
- `GET /classes` - 支援的藥物類別
- `POST /detect` - 從 URL 檢測藥丸
- `POST /detect-file` - 從上傳檔案檢測藥丸  
- `GET /test` - 網頁測試介面

## 📊 技術規格

- **架構**: RF-DETR (即時檢測變換器)
- **輸入尺寸**: 560x560 像素
- **格式**: ONNX 跨平台推理
- **預處理**: ImageNet 正規化
- **後處理**: 信心度過濾 + Top-K 選擇

## 📜 更多資訊

- 詳細 API 文檔：[README_API.md](README_API.md)
