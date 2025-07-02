# 藥丸檢測系統技術演進

## 🎯 演進總覽

| 階段 | 技術棧 | 核心特點 | 結果 |
|------|--------|----------|------|
| **原始實現** | .pth + PyTorch | 官方標準，依賴重 | 研究驗證 |
| **中間態1** | .onnx + OpenCV | 預處理順序錯誤 | 精度問題 |
| **中間態2** | .onnx + Pure Math | 完美精度，極慢 | 技術驗證 |
| **最終方案** | .onnx + Pillow | 修正操作順序 | 生產版本 |

## 🔑 核心洞察

### 1. 預處理操作順序問題

❌ **錯誤順序** (中間態1)：
```
image → to_tensor → normalize → resize
```
- 在標準化數據上 resize 導致精度損失

✅ **採用方案** (最終實現)：
```
image → resize → to_tensor → normalize
```
- 在原始像素域 (uint8 [0,255]) resize，避免精度損失

### 2. 後處理算法優化

❌ **複雜方案** (中間態1)：
```
全域 Top-K 搜索 → 可能同位置多檢測 → 複雜索引計算
```
- 在所有 300×6=1800 個值中搜索 Top-30
- 可能同一位置檢測到多種藥物

✅ **採用方案** (最終實現)：
```
每位置最高分類 → 閾值過濾 → Top-K 選擇
```
- 避免同位置重複檢測，符合物理直覺
- 算法更清晰，邏輯更直觀

### 技術對比

| 方案 | 預處理 | 後處理 | 依賴 | 性能 | 複雜度 |
|------|--------|--------|------|------|--------|
| Legacy | `to_tensor→normalize→resize` (OpenCV) | 全域 Top-K 搜索 | 重 | 中等 | 高 |
| Pure Math | 手動實現所有算法 | 完全數學實現 | 極輕 | 極慢 | 極高 |
| 採用方案 | `resize→to_tensor→normalize` (PIL) | 每位置最高分類 | 輕 | 快 | 低 |

## 🏗️ 最終架構

```
/workspace/
├── main.py                    # 🚀 主應用入口
├── uvicorn.prod.py           # 🏭 生產環境啟動
├── requirements.txt          # 📦 輕量依賴清單
│
├── app/                      # 📂 核心模組
│   ├── pill_detector.py     # 🎯 檢測器核心
│   ├── config.py            # ⚙️ 配置管理
│   └── _annotations.coco.json # 📊 類別定義
│
├── models/                   # 🧠 ONNX 模型
│   └── inference_model.onnx
│
├── legacy/                   # 📦 演進歷史
│   └── main_legacy_opencv.py # 中間態1完整實現
│
├── tests/                    # 🧪 測試驗證
├── docs/                     # 📚 技術文檔
└── scripts/                  # 🔧 工具腳本
```

## 🔧 主要實現

- **解決精度問題**：調整預處理操作順序
- **改進後處理邏輯**：避免同位置多檢測
- **精簡依賴**：從 PyTorch 生態轉為 numpy + Pillow
- **模組化設計**：代碼結構清晰