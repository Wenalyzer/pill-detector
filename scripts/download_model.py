#!/usr/bin/env python3
import os
import requests

def download_model():
    """從 GitHub Release 下載 ONNX 模型"""
    
    GITHUB_USER = "Wenalyzer"
    REPO_NAME = "pill-detector"
    VERSION = "v3.0.1"
    FILENAME = "inference_model_v3.0.1.onnx"
    
    MODEL_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}/releases/download/{VERSION}/{FILENAME}"
    MODEL_PATH = "models/inference_model_v3.0.1.onnx"
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ 模型檔案已存在: {MODEL_PATH}")
        return True
    
    print(f"📥 從 GitHub Release 下載模型...")
    print(f"🔗 URL: {MODEL_URL}")
    
    os.makedirs("models", exist_ok=True)
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📊 下載進度: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
        
        print(f"\n✅ 模型下載完成: {MODEL_PATH}")
        print(f"📏 檔案大小: {os.path.getsize(MODEL_PATH)/1024/1024:.1f}MB")
        return True
        
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        exit(1)