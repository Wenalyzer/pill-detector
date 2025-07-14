# 💊 藥丸檢測 API 

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.13-green.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.22.0-orange.svg)](https://onnxruntime.ai/)
[![Pillow](https://img.shields.io/badge/Pillow-11.2.1-blue.svg)](https://pillow.readthedocs.io/)
[![uvicorn](https://img.shields.io/badge/uvicorn-0.34.3-green.svg)](https://www.uvicorn.org/)

使用 RF-DETR ONNX 模型進行藥品識別的 FastAPI 應用程式。

## 🔧 主要特點

- ✅ **精簡依賴**: 9個核心套件，無需 PyTorch/OpenCV
- ✅ **完整 API**: 5個端點涵蓋檢測、查詢、測試功能
- ✅ **完整應用**: FastAPI 應用，包含健康檢查和測試介面
- ✅ **容器化**: Docker 支援

## 🏥 支援的藥物類別

總計 **50種** 常見藥物，支援中英文標註：

- **愛舒疼錠** (ACETAL) 
- **愛達信錠** (ALDACTIN)
- **安邦錠** (ALPRAZOLAM)
- **安莫西林膠囊** (AMOXICILLIN)
- **安西林膠囊** (ANCILLIN)
- **美克糖錠** (ANKOMIN)
- **妙化錠** (BIOTASE)
- **普疏痛錠** (BONSTAN)
- **百痛寧錠** (BYTONLIN)
- **服敏錠** (CERIN)
- **氯芬尼拉明錠** (CHLORPHENIRAMINE)
- **希美廸錠** (CIMETIDINE)
- **賜腦清錠** (CINNAZINE)
- **可拿平錠** (CONAPIN)
- **喜達鎮錠** (CYTADINE)
- **特釋敏錠** (DIMETINE)
- **得安穩錠160mg** (DIOVAN_160mg)
- **得安穩錠80mg** (DIOVAN_80mg)
- **敵芬尼朵錠** (DIPHENIDOL)
- **二羥丙基茶生僉錠** (DIPROPHYLLIN)
- **得喜寧膠囊** (DOXYNIN)
- **抑克敏錠** (ECOMIN)
- **健安心錠** (ENTRESTO)
- **葉酸錠** (FOLACIN)
- **福適佳錠** (FORXIGA)
- **理冒伯樂錠** (FUCOLE PARAN)
- **去咳寧膠囊** (FUCOU)
- **克瀉寧錠** (K.B.T.)
- **胃全膠囊** (LANSOPRAZOLE)
- **立可隆錠** (LICOLONE)
- **那寶穩錠** (LOSA_HYDRO)
- **莫克寧錠** (MOXICLAV)
- **胃默適錠** (MOZAPRY)
- **脈優錠** (NORVASC)
- **痙得寧錠** (NUSPAS)
- **咳祿錠** (OXOLA)
- **普拿疼伏冒加強錠** (PANADOL COLD EXTRA)
- **愛舒可羅錠** (REGROW)
- **明德銳利錠** (RELAX ANALGESIC)
- **禮痛保膠囊** (RELECOX)
- **嗽隨康錠** (SAUSCON)
- **悠然錠** (SILENCE)
- **舒敏錠** (SOMIN)
- **泰克胃通錠** (TAKEPRON)
- **便通樂錠** (THROUGH)
- **力停疼錠** (TINTEN)
- **痛能錠** (TOLAX)
- **優佳朗錠** (UCALON)
- **立除痛錠** (UTRAPHEN)
- **胃之盟錠** (WEITUMON) 

## 🚀 快速啟動

### 模型準備
```bash
# 下載 ONNX 模型 (二選一)
python scripts/download_model.py

# 或從 GitHub Releases 手動下載
# https://github.com/Wenalyzer/pill-detector/releases
```

### 本地開發
```bash
# 開發模式啟動 (自動重載，修改代碼即時生效)
python main.py

# 或使用 uvicorn 命令行啟動
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 生產環境
```bash
# 使用生產配置啟動 (支援環境變數 PORT)
python uvicorn.prod.py

# 或使用 Docker
docker build -t pill-detector .
docker run -d -p 8000:8000 pill-detector

# 雲端部署時可指定端口
PORT=8080 python uvicorn.prod.py
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
│   ├── config.py            # ⚙️ 統一配置管理
│   ├── detection_service.py # 🎮 檢測服務業務邏輯層
│   ├── pill_detector.py     # 🎯 核心檢測器（模型推理）
│   ├── image_annotator.py   # 🖼️ 圖像標註器
│   ├── utils/               # 🔧 工具模組
│   │   ├── coordinate_utils.py  # 座標轉換工具
│   │   └── font_utils.py        # 字體處理工具
│   └── _annotations.coco.json # 📊 COCO格式類別定義
│
├── models/                   # 🧠 ONNX 模型
│   └── inference_model_v3.0.1.onnx  # (python scripts/download_model.py 下載)
│
├── tests/                    # 🧪 測試與驗證
│   ├── test_api.py          # API 功能測試
│   ├── image.jpg            # 測試圖片
│   └── IMG_*.jpg            # 額外測試圖片
│
├── docs/                     # 📚 技術文檔
│   ├── guides/              # 使用指南
│   ├── architecture/        # 系統架構圖
│   └── specs/               # 技術規格
│
├── legacy/                   # 📦 演進歷史保存
│   ├── main_legacy_opencv.py    # OpenCV 版本實現
│   ├── elegant_solution_spec.md  # Pillow 實現規格
│   └── pure_math_spec.md        # 純數學方案規格
│
├── scripts/                  # 🔧 工具腳本
│   └── download_model.py     # 模型下載腳本
│
└── .github/workflows/        # 🔄 CI/CD 配置
    ├── smart-build.yml       # GitHub Actions 建置流程
    └── deploy.yml            # 部署流程配置
```

## 🔗 API 端點

### 📋 端點總覽
- `GET /` - API 狀態和基本資訊
- `GET /health` - 健康檢查與服務狀態
- `GET /classes` - 取得所有支援的藥物類別 (中英文對照)
- `POST /detect` - 統一檢測端點 (支援檔案上傳和 URL)
- `GET /test` - 網頁測試介面

## 🎭 技術演進故事

### 目標
使用 .onnx 輕量化重現 .pth 的推理結果，用 onnxruntime + Pillow 取代原始 RF-DETR 的 predict 方法。

### 關鍵洞察：操作順序的重要性

❌ **問題方法**: `to_tensor → normalize → resize`
- 在標準化數據上 resize 導致精度損失

✅ **採用方案**: `resize → to_tensor → normalize`  
- 在原始像素域 (uint8 [0,255]) resize，保持精度

### 後處理優化

❌ **複雜方法**: 全域 Top-K 搜索 → 可能同位置多檢測  
✅ **採用方案**: 每位置最高分類 → 避免重複檢測

### 最終成果

- 🪶 **依賴精簡**: torch + torchvision + supervision → **numpy + Pillow**
- 🔧 **API 設計**: 統一 /detect 端點
- 📊 **架構實現**: 模組化設計

## 📚 文檔導引

### 📖 使用指南
- **[API 使用指南](docs/guides/API_GUIDE.md)** - 完整的端點使用說明和範例
- **[技術演進歷程](docs/guides/TECHNICAL_JOURNEY_COMPACT.md)** - 從概念到實現的技術故事

### 🏗️ 系統架構
- **[系統架構圖](docs/architecture/request_flow.png)** - uvicorn + FastAPI + 自訂服務的協作流程
- **[專案技術總結](docs/specs/PROJECT_SUMMARY.md)** - 完整的架構設計和實現細節

### 🔧 技術規格 (歷史保存)
- **[Pillow 實現規格](legacy/elegant_solution_spec.md)** - 移除 OpenCV 的實現方案
- **[純數學方案規格](legacy/pure_math_spec.md)** - 純 numpy 數學運算方案

### 🚀 建置與部署
- **[Docker 配置](Dockerfile)** - 容器化部署配置
- **[CI/CD 流程](.github/workflows/smart-build.yml)** - GitHub Actions 自動建置
