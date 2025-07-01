# 💊 藥丸檢測 API 

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

使用 RF-DETR ONNX 模型進行藥品識別的 FastAPI 應用程式，採用優雅的 **resize → to_tensor → normalize** 預處理流程。

**🎯 技術成果**: 從 PyTorch 重型框架轉為輕量化 ONNX 方案，移除 OpenCV 依賴，性能大幅提升

## 🎯 核心特色

- ✅ **輕量依賴**: 9個核心包，無需 PyTorch/OpenCV 重型框架
- ✅ **優雅實現**: 在原始像素域操作，避免精度損失  
- ✅ **統一端點**: 單一 `/detect` 支援 URL 和檔案上傳
- ✅ **生產就緒**: 完整的 FastAPI 應用，包含健康檢查和測試介面
- ✅ **高性能**: 預處理 ~6ms，總檢測 ~370ms *
- ✅ **容器化**: Docker 支援，適合雲端部署

> **性能基準**: Intel i5-12600KF (6P 4E 16T) + 48GB RAM + Docker 環境

## 🏥 支援的藥物類別

- **Amoxicillin** 
- **Diovan 160mg** 
- **Lansoprazole** 
- **Relecox** 
- **Takepron** 
- **Utraphen** 

## 🚀 快速啟動

### 本地開發
```bash
# 直接啟動主應用
python main.py

# 或使用 uvicorn 手動啟動
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 生產環境
```bash
# 使用優化配置啟動
python uvicorn.prod.py

# 或使用 Docker
docker build -t pill-detector .
docker run -d -p 8000:8000 pill-detector
```

### 驗證部署
```bash
# 檢查服務狀態
curl http://localhost:8000/health

# 訪問測試介面
open http://localhost:8000/test
```

## 📁 專案結構

```
/workspace/
├── main.py                    # 🚀 主應用入口 (FastAPI)
├── uvicorn.prod.py           # 🏭 生產環境配置
├── requirements.txt          # 📦 輕量依賴清單
├── Dockerfile                # 🐳 Docker 建置配置
│
├── app/                      # 📂 核心應用模組
│   ├── __init__.py          # 📦 模組初始化
│   ├── pill_detector.py     # 🎯 檢測器核心 (優雅方案)
│   ├── config.py            # ⚙️ 配置管理
│   └── _annotations.coco.json # 📊 類別定義
│
├── models/                   # 🧠 ONNX 模型
│   └── inference_model.onnx  # (自動下載)
│
├── tests/                    # 🧪 測試與驗證
│   ├── test_api.py          # API 功能測試
│   ├── compare_detection_methods.py # 檢測方法比較
│   └── image.jpg            # 測試圖片
│
├── docs/                     # 📚 技術文檔
│   ├── 01_TECHNICAL_JOURNEY_COMPACT.md # 技術演進歷程
│   └── 02_API_GUIDE.md      # API 使用指南
│
├── legacy/                   # 📦 演進歷史保存
│   ├── main_legacy_opencv.py    # OpenCV 版本實現
│   ├── elegant_solution_spec.md  # 優雅方案規格
│   ├── pure_math_spec.md        # 純數學方案規格
│   └── rfdetr_original_spec.md   # 原始 RF-DETR 規格
│
├── scripts/                  # 🔧 工具腳本
│   └── download_model.py     # 模型下載腳本
│
└── .github/workflows/        # 🔄 CI/CD 配置
    └── smart-build.yml       # GitHub Actions 建置流程
```

## 🔗 API 使用

### ⚡ 快速開始
```bash
# 🚀 推薦：檔案上傳 (~0.2s)
curl -X POST "https://your-api.run.app/detect" \
  -F "file=@image.jpg"

# 📥 備用：URL 下載 (~2.5s)  
curl -X POST "https://your-api.run.app/detect" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```

### 📋 端點總覽
- `POST /detect` - 統一檢測端點 (檔案上傳 15 倍速度優勢)
- `GET /health` - 健康檢查
- `GET /test` - 網頁測試介面

> 📖 **詳細使用說明**: [`docs/02_API_GUIDE.md`](docs/02_API_GUIDE.md)

## 🎭 技術演進故事

### 目標
使用 .onnx 輕量化重現 .pth 的推理結果，用 onnxruntime + Pillow 取代原始 RF-DETR 的 predict 方法。

### 關鍵洞察：操作順序的重要性

❌ **問題方法**: `to_tensor → normalize → resize`
- 在標準化數據上 resize 導致精度損失

✅ **優雅方案**: `resize → to_tensor → normalize`  
- 在原始像素域 (uint8 [0,255]) resize，完美保持精度

### 後處理優化

❌ **複雜方法**: 全域 Top-K 搜索 → 可能同位置多檢測  
✅ **優雅方案**: 每位置最高分類 → 避免重複檢測

### 最終成果

- 🪶 **依賴精簡**: torch + torchvision + supervision → **numpy + Pillow**
- 🔧 **API 簡化**: 分離的端點 → **統一 /detect 端點**
- 🎨 **代碼優化**: 複雜後處理 → **清晰邏輯**
- 📊 **架構改進**: 單體設計 → **模組化架構**

## 📚 相關文件

### 使用指南
- **🚀 技術演進歷程**: [`docs/01_TECHNICAL_JOURNEY_COMPACT.md`](docs/01_TECHNICAL_JOURNEY_COMPACT.md)
- **📖 API 使用指南**: [`docs/02_API_GUIDE.md`](docs/02_API_GUIDE.md)

### 技術規格 (歷史保存)
- **✨ 優雅方案規格**: [`legacy/elegant_solution_spec.md`](legacy/elegant_solution_spec.md)
- **🧮 純數學方案規格**: [`legacy/pure_math_spec.md`](legacy/pure_math_spec.md)  
- **🎯 原始 RF-DETR 規格**: [`legacy/rfdetr_original_spec.md`](legacy/rfdetr_original_spec.md)

### 建置與部署
- **🐳 Docker 配置**: [`Dockerfile`](Dockerfile)
- **🔄 CI/CD 流程**: [`.github/workflows/smart-build.yml`](.github/workflows/smart-build.yml)

## 🧪 測試驗證

### 功能測試
```bash
# API 功能測試
python tests/test_api.py

# 檢測方法比較
python tests/compare_detection_methods.py
```

### 本地測試
```bash
# 啟動服務
python main.py

# 訪問測試介面
open http://localhost:8000/test

# 檢查健康狀態
curl http://localhost:8000/health
```

### 性能基準
測試環境：**Intel i5-12600KF (6P 4E 16T) + 48GB RAM + Docker**
- 🚀 初始化時間: ~1.2s (冷啟動，包含模型載入)
- ⚡ 預處理時間: ~6ms (平均10次)
- 🎯 完整檢測: ~370ms (包含後處理和標註)

> 💡 不同硬體環境性能會有差異，建議在目標部署環境測試