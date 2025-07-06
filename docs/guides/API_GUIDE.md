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
