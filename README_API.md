# 藥丸檢測 API 使用說明

## 1. 服務簡介
本 API 提供藥丸影像檢測功能，使用 RF-DETR 模型進行物件檢測，支援圖片 URL 和檔案上傳兩種方式。

## 2. API 端點

### 基礎資訊
- **服務網址**: `https://pill-detector-23010935669.us-central1.run.app`
- **健康檢查**: `GET /health`
- **支援類別**: `GET /classes`

### 檢測端點

#### 🌐 URL 檢測
- **端點**: `POST /detect`
- **Content-Type**: `application/json`

#### 📁 檔案上傳檢測  
- **端點**: `POST /detect-file`
- **Content-Type**: `multipart/form-data`

## 3. 請求格式

### URL 檢測
```json
{
  "image_url": "https://example.com/pill-image.jpg"
}
```

### 檔案上傳
```bash
curl -X POST "https://pill-detector-23010935669.us-central1.run.app/detect-file" \
  -F "file=@your-image.jpg"
```

## 4. 回應格式
```json
{
  "success": true,
  "message": "檢測完成，發現 2 個藥丸",
  "data": {
    "detections": [
      {
        "class_id": 1,
        "class_name": "Amoxicillin",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "total_detections": 2
  }
}
```

## 5. 回應欄位說明

- **`success`**: 檢測是否成功 (boolean)
- **`message`**: 結果訊息 (string)  
- **`data.detections`**: 檢測結果陣列
  - `class_id`: 類別 ID (integer)
  - `class_name`: 藥丸名稱 (string)
  - `confidence`: 信心度 0-1 (float)
  - `bbox`: 邊界框座標 [x1, y1, x2, y2] (array)
- **`data.annotated_image`**: 標註後圖片 (base64 data URL)
- **`data.total_detections`**: 檢測到的藥丸總數 (integer)

### Python 處理範例（含上傳到 GCP/AWS 雲端）

#### 1. 取得 API 回應並儲存圖片
```python
import requests
import base64

url = "https://pill-detector-23010935669.us-central1.run.app/detect"
payload = {
    "image_url": "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg",
    "threshold": 0.5
}
resp = requests.post(url, json=payload)
result = resp.json()

if result.get("success") and result.get("annotated_image_base64"):
    # 轉成二進位圖片
    img_bytes = base64.b64decode(result["annotated_image_base64"])
    # 儲存本地檔案（可選）
    with open("annotated.jpg", "wb") as f:
        f.write(img_bytes)
else:
    print("API 回傳失敗:", result)
```

#### 2. 上傳到 Google Cloud Storage (GCS)
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('你的bucket名稱')
blob = bucket.blob('路徑/annotated.jpg')
blob.upload_from_string(img_bytes, content_type='image/jpeg')
print('GCS 圖片網址:', blob.public_url)
```

#### 3. 上傳到 AWS S3
```python
import boto3
import io

s3 = boto3.client('s3')
s3.upload_fileobj(
    Fileobj=io.BytesIO(img_bytes),
    Bucket='你的bucket名稱',
    Key='路徑/annotated.jpg',
    ExtraArgs={'ContentType': 'image/jpeg', 'ACL': 'public-read'}
)
url = f'https://{"你的bucket名稱"}.s3.amazonaws.com/路徑/annotated.jpg'
print('S3 圖片網址:', url)
```

> 請先安裝對應套件：`pip install google-cloud-storage boto3`
> 並依官方文件設好認證與權限。

### 前端顯示標註圖片

將 `annotated_image_base64` 用於 `<img>` 標籤：
```html
<img src="data:image/jpeg;base64,{{ annotated_image_base64 }}" />
```
