# 💊 Pill Detector

## 🚀 線上體驗

直接打開網頁，選擇藥物圖片上傳即可辨識：

👉 [https://pill-detector-23010935669.us-central1.run.app/test](https://pill-detector-23010935669.us-central1.run.app/test)

目前支援 6 種藥物：

- Amoxicillin
- Diovan 160mg
- Lansoprazole
- Relecox
- Takepron
- Utraphen

（如需更新藥物名稱，請參考 `class_names` 或模型設定）

---

## 🐳 Docker 使用方式

1. 先拉取 image：

   ```
   docker pull ghcr.io/wenalyzer/pill-detector:latest
   ```

2. 啟動服務（預設監聽 8000 port）：

   ```
   docker run -d -p 8000:8000 ghcr.io/wenalyzer/pill-detector:latest
   ```

3. 用瀏覽器打開：

   ```
   http://localhost:8000/test
   ```

4. 上傳藥物圖片即可辨識！
