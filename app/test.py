import requests
import json
import time
import base64
from PIL import Image
import io

def test_api_comprehensive():
    """全面測試 API"""
    
    base_url = "https://pill-detector-23010935669.us-central1.run.app" # 替換為你的 API URL
    
    print("🚀 開始測試藥丸辨識 API")
    print("=" * 50)
    
    # 1. 測試連接
    print("1️⃣ 測試 API 連接...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ API 運行中: {response.json()}")
        else:
            print(f"   ❌ API 無回應: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ 連接失敗: {e}")
        return
    
    # 2. 測試健康檢查
    print("\n2️⃣ 測試健康檢查...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"   ✅ 健康狀態: {health_data}")
        
        if not health_data.get("model_loaded", False):
            print("   ⚠️ 模型未載入，請檢查模型檔案")
            return
    except Exception as e:
        print(f"   ❌ 健康檢查失敗: {e}")
        return
    
    # 3. 測試偵測功能
    print("\n3️⃣ 測試藥丸偵測...")
    
    # 使用測試圖片 URL
    test_image_urls = [
        "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg"
    ]
    
    for i, image_url in enumerate(test_image_urls):
        print(f"\n   📸 測試圖片 {i+1}: {image_url}")
        
        try:
            start_time = time.time()
            
            test_data = {
                "image_url": image_url,
                "threshold": 0.5
            }
            
            response = requests.post(
                f"{base_url}/detect",
                json=test_data,
                timeout=60
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 偵測成功!")
                print(f"   📊 偵測到 {len(result['detections'])} 個物件")
                print(f"   ⚡ 推論時間: {result['inference_time_ms']}ms")
                print(f"   🌐 總請求時間: {total_time*1000:.1f}ms")
                
                # 顯示偵測結果
                for detection in result['detections']:
                    print(f"      🔍 Pill {detection['pill_name']}: "
                          f"信心度 {detection['confidence']:.3f}")
                
                # 可選：儲存標註圖片
                if result.get('annotated_image_base64'):
                    save_annotated_image(result['annotated_image_base64'], f"result_{i+1}.jpg")
                    
            else:
                print(f"   ❌ 偵測失敗: {response.status_code}")
                print(f"   📄 錯誤訊息: {response.text}")
                
        except Exception as e:
            print(f"   ❌ 請求失敗: {e}")
    
    print(f"\n✅ API 測試完成")

def save_annotated_image(base64_string, filename):
    """儲存 base64 圖片"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image.save(filename)
        print(f"   💾 標註圖片已儲存: {filename}")
    except Exception as e:
        print(f"   ⚠️ 儲存圖片失敗: {e}")

if __name__ == "__main__":
    test_api_comprehensive()