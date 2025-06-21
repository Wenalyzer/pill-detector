# 藥丸辨識 API 使用說明

## 1. 服務簡介
本 API 提供藥丸影像辨識功能，支援圖片網址輸入，回傳辨識結果與標註後圖片。

## 2. API 端點
- URL：`https://pill-detector-23010935669.us-central1.run.app/detect`
- 方法：POST
- Content-Type：application/json

## 3. 請求格式
```json
{
  "image_url": "圖片網址 (string)",
  "threshold": 0.5 // 信心度閾值，選填，預設 0.5
}
```

## 4. 回應格式
```json
{
  "success": true,
  "detections": [
    {
      "pill_name": "藥丸名稱",
      "confidence": 0.98,
      "bbox": [x1, y1, x2, y2]
    }
    // ...
  ],
  "annotated_image_base64": "base64字串（標註後圖片）",
  "inference_time_ms": 123.45,
  "total_detections": 2
}
```

## 5. 如何處理 API 回應

API 會回傳一個 JSON 物件，主要欄位如下：
- `success`：是否成功
- `detections`：偵測到的藥丸清單（每個包含名稱、信心度、座標等）
- `annotated_image_base64`：標註後圖片的 base64 字串
- `inference_time_ms`：推論時間（毫秒）
- `total_detections`：偵測到的藥丸數量

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
