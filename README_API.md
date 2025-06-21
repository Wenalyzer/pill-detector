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

## 5. 範例
### Python
```python
import requests
url = "https://pill-detector-23010935669.us-central1.run.app/detect"
payload = {
    "image_url": "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg",
    "threshold": 0.5
}
resp = requests.post(url, json=payload)
print(resp.json())
```

### curl
```bash
curl -X POST "https://pill-detector-23010935669.us-central1.run.app/detect" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg", "threshold": 0.5}'
```

## 6. 注意事項
- `image_url` 必須為公開可存取的圖片網址。
- `threshold` 越高，檢測越嚴格（建議 0.3~0.7）。
- 回傳的 `annotated_image_base64` 可直接顯示於 `<img src="data:image/jpeg;base64,..." />`。
- 若有錯誤，會回傳 HTTP 4xx/5xx 及 `detail` 欄位。
