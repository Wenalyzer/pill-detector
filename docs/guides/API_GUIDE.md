# 💊 藥丸檢測 API 使用指南

## 🎯 服務簡介
本 API 提供藥丸影像檢測功能，使用 ONNX Runtime 進行推理。

## 🔧 使用方式

### 檔案上傳
```bash
curl -X POST "https://pill-detector-23010935669.us-central1.run.app/detect" \
  -F "file=@image.jpg"
```

### URL 下載
```bash
curl -X POST "https://pill-detector-23010935669.us-central1.run.app/detect" \
  -F "image_url=https://example.com/image.jpg"
```

## 💡 團隊整合建議

### LINE Bot 開發者
```python
# 直接傳遞圖片位元組
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 從 LINE 取得圖片
    content = line_bot_api.get_message_content(event.message.id)
    image_bytes = content.content
    
    # 直接 POST 到檢測 API
    response = requests.post(
        "https://your-api.run.app/detect",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")}
    )
    result = response.json()
```

### 網頁應用開發者
```javascript
// 使用 FormData 上傳檔案
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/detect', {
    method: 'POST',
    body: formData
}).then(response => response.json());
```

### 行動應用開發者
```python
# 上傳相機拍攝的圖片
import requests

with open('camera_photo.jpg', 'rb') as f:
    response = requests.post(
        'https://your-api.run.app/detect',
        files={'file': f}
    )
```

## 🔗 API 端點總覽

### 📋 基礎資訊端點
- **API 狀態**: `GET /` - 顯示 API 資訊
- **健康檢查**: `GET /health` - 服務健康狀態
- **藥物類別**: `GET /classes` - 取得所有支援的藥物類別（中英文對照）

### 檢測端點

#### 🎯 統一檢測端點
- **端點**: `POST /detect`
- **支援方式**: 檔案上傳和 URL 檢測

## 📊 使用方式對比

| 方法 | 適用場景 |
|------|----------|
| 檔案上傳 | LINE Bot、網頁上傳、行動應用 |
| URL 下載 | 無法直接取得檔案的場景 |

## 📝 請求格式

### 檔案上傳
```bash
curl -X POST "/detect" \
  -F "file=@your-image.jpg"
```

### URL 檢測
```bash
curl -X POST "/detect" \
  -F "image_url=https://example.com/pill-image.jpg"
```

## 📋 回應格式
```json
{
  "success": true,
  "message": "檢測完成，發現 2 個藥丸",
  "data": {
    "detections": [
      {
        "class_id": 1,
        "class_name": "安莫西林膠囊",
        "class_name_en": "Amoxicillin",
        "class_name_zh": "安莫西林膠囊",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "total_detections": 2,
    "image_info": {
      "original_size": [1920, 1080],
      "mode": "RGB"
    },
    "quality_analysis": {
      "should_retake": false,
      "reason": "good_quality",
      "message": "檢測品質良好，識別結果可信",
      "quality_score": 0.95,
      "suggestions": ["可考慮重新拍攝以提高識別準確度"],
      "uncertain_items": []
    }
  }
}
```

## 📖 回應欄位說明

- **`success`**: 檢測是否成功 (boolean)
- **`message`**: 結果訊息 (string)  
- **`data.detections`**: 檢測結果陣列
  - `class_id`: 類別 ID (integer)
  - `class_name`: 中文藥丸名稱 (string)
  - `class_name_en`: 英文藥丸名稱 (string)
  - `class_name_zh`: 中文藥丸名稱 (string)
  - `confidence`: 信心度 0-1 (float)
  - `bbox`: 邊界框座標 [x1, y1, x2, y2] (array)
- **`data.annotated_image`**: 標註後圖片 (base64 data URL)
- **`data.total_detections`**: 檢測到的藥丸總數 (integer)
- **`data.image_info`**: 圖片資訊 (object)
  - `original_size`: 原始圖片尺寸 [width, height] (array)
  - `mode`: 圖片模式，如 "RGB" (string)
- **`data.quality_analysis`**: 🆕 檢測品質分析 (object)
  - `should_retake`: 是否建議重新拍攝 (boolean)
  - `reason`: 分析原因 ("good_quality", "low_confidence", "no_detection", "partial_uncertainty") (string)
  - `message`: 品質分析訊息 (string)
  - `quality_score`: 品質分數 0-1，越高越好 (float, optional)
  - `suggestions`: 改善建議陣列 (array, optional)
  - `uncertain_items`: 低信心度項目索引 (array, optional)

## 💻 程式整合範例

### Python 範例
```python
import requests
import base64

# 檔案上傳
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'https://your-api.run.app/detect',
        files={'file': f}
    )

result = response.json()
if result['success']:
    print(f"檢測到 {result['data']['total_detections']} 個藥丸")
    
    # 處理檢測結果
    for detection in result['data']['detections']:
        print(f"發現: {detection['class_name']} ({detection['class_name_en']})")
        print(f"信心度: {detection['confidence']:.2f}")
        print(f"位置: {detection['bbox']}")
    
    # 處理標註圖片
    base64_image = result['data']['annotated_image']
    # 可直接用於前端顯示
```

### JavaScript 範例
```javascript
// 檔案上傳
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log(`發現 ${data.data.total_detections} 個藥丸`);
        
        // 顯示檢測結果
        data.data.detections.forEach((detection, index) => {
            console.log(`${index + 1}. ${detection.class_name} (${detection.class_name_en})`);
            console.log(`   信心度: ${detection.confidence.toFixed(2)}`);
        });
        
        // 🆕 檢查品質分析並提醒用戶
        if (data.data.quality_analysis && data.data.quality_analysis.should_retake) {
            const qa = data.data.quality_analysis;
            alert(`建議重新拍攝：${qa.message}\n\n改善建議：\n${qa.suggestions?.join('\n• ')}`);
        }
        
        // 顯示標註圖片
        document.getElementById('result').src = data.data.annotated_image;
    }
});
```

### HTML 顯示
```html
<img id="result" src="" alt="檢測結果" />
```

### 📋 藥物類別端點
```bash
curl -X GET "https://pill-detector-23010935669.us-central1.run.app/classes"
```

**回應格式：**
```json
{
  "classes": [
    {
      "english": "Amoxicillin",
      "chinese": "安莫西林膠囊"
    },
    {
      "english": "Diovan_160mg", 
      "chinese": "得安穩錠160mg"
    }
  ],
  "total_classes": 18
}
```

## 🆕 智能品質分析功能

### 📊 **功能概述**
API 現在會自動分析每次檢測的品質，並在檢測結果不理想時建議用戶重新拍攝。

### 🔍 **分析項目**
1. **信心度檢查**: 檢測低信心度的藥丸識別
2. **檢測完整性**: 確認是否有遺漏的藥丸
3. **品質評分**: 提供 0-1 的整體品質分數

### 📋 **品質分析回應詳解**

#### **品質良好** (`reason: "good_quality"`)
```json
"quality_analysis": {
  "should_retake": false,
  "reason": "good_quality", 
  "message": "檢測品質良好，識別結果可信",
  "quality_score": 0.95
}
```

#### **部分不確定** (`reason: "partial_uncertainty"`)
```json
"quality_analysis": {
  "should_retake": false,
  "reason": "partial_uncertainty",
  "message": "檢測品質良好，但有 2 個藥丸的信心度較低", 
  "suggestions": ["可考慮重新拍攝以提高識別準確度"],
  "uncertain_items": [3, 5],
  "quality_score": 0.75
}
```

#### **建議重拍** (`reason: "low_confidence"`)
```json
"quality_analysis": {
  "should_retake": true,
  "reason": "low_confidence",
  "message": "有 3 個藥丸的識別信心度較低，建議重新拍攝",
  "suggestions": [
    "確保光線充足，避免陰影遮擋",
    "將手機靠近一些，讓藥丸更清晰", 
    "確保藥丸表面清潔，字體清晰可見",
    "避免手震，保持拍攝穩定"
  ],
  "uncertain_items": [1, 2, 4],
  "quality_score": 0.45
}
```

#### **未檢測到** (`reason: "no_detection"`)
```json
"quality_analysis": {
  "should_retake": true,
  "reason": "no_detection",
  "message": "未檢測到任何藥丸，建議重新拍攝",
  "suggestions": [
    "確保藥丸清晰可見",
    "改善光線條件", 
    "調整拍攝角度或距離"
  ]
}
```

### 💡 **前端整合建議**

#### **JavaScript 範例**
```javascript
.then(data => {
    if (data.success) {
        // 顯示檢測結果
        displayDetections(data.data.detections);
        
        // 檢查品質分析
        const qa = data.data.quality_analysis;
        if (qa.should_retake) {
            // 顯示重拍建議
            showRetakeDialog(qa);
        } else if (qa.reason === 'partial_uncertainty') {
            // 顯示可選重拍提示
            showOptionalRetakeHint(qa);
        }
    }
});

function showRetakeDialog(qa) {
    const message = `
        📷 建議重新拍攝
        
        ${qa.message}
        
        💡 改善建議：
        ${qa.suggestions.map(s => `• ${s}`).join('\n')}
        
        是否重新拍攝？
    `;
    
    if (confirm(message)) {
        // 重新啟動拍照功能
        startCamera();
    }
}
```

#### **品質分數使用**
```javascript
const qa = data.data.quality_analysis;
const qualityPercent = Math.round(qa.quality_score * 100);

// 顯示品質指示器
document.getElementById('quality-indicator').innerHTML = `
    <div class="quality-score ${qa.quality_score > 0.8 ? 'good' : qa.quality_score > 0.6 ? 'fair' : 'poor'}">
        品質: ${qualityPercent}%
    </div>
`;
```

### 🎯 **最佳實踐**
1. **強制重拍**: `should_retake: true` 時，建議用戶重新拍攝
2. **可選提示**: `partial_uncertainty` 時，可提供可選的重拍選項
3. **品質指示**: 使用 `quality_score` 提供視覺化的品質指示器
4. **具體建議**: 顯示 `suggestions` 陣列中的具體改善建議
