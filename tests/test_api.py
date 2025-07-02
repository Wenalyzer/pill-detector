import requests
import json
import time
import base64
from PIL import Image
import io
import os

def test_api_comprehensive():
    """全面測試藥丸檢測 API"""
    
    # 使用本地端點進行測試
    base_url = "http://localhost:8000"
    
    print("🚀 開始測試藥丸檢測 API")
    print("=" * 50)
    
    # 追蹤測試結果
    test_results = {
        'connection': False,
        'health': False,
        'file_upload': False,
        'url_detection': False,
        'error_handling': False
    }
    
    # 1. 測試連接
    print("1️⃣ 測試 API 連接...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ API 運行中: {response.json()}")
            test_results['connection'] = True
        else:
            print(f"   ❌ API 無回應: {response.status_code}")
            assert False, f"API 連接失敗，狀態碼: {response.status_code}"
    except Exception as e:
        print(f"   ❌ 連接失敗: {e}")
        print(f"   💡 提示: 請先啟動 API 服務 (python main.py)")
        assert False, f"API 連接異常: {e}"
    
    # 2. 測試健康檢查
    print("\n2️⃣ 測試健康檢查...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"   ✅ 健康狀態: {health_data}")
        
        if not health_data.get("service_ready", False):
            print("   ⚠️ 服務未就緒，請檢查模型檔案和初始化狀態")
            print("   💡 提示: 執行 python scripts/download_model.py 下載模型")
            assert False, f"服務未就緒: {health_data}"
        
        test_results['health'] = True
    except Exception as e:
        print(f"   ❌ 健康檢查失敗: {e}")
        assert False, f"健康檢查異常: {e}"
    
    # 3. 測試檔案上傳檢測
    print("\n3️⃣ 測試檔案上傳檢測...")
    test_image_path = "tests/image.jpg"
    
    if os.path.exists(test_image_path):
        try:
            start_time = time.perf_counter()
            
            with open(test_image_path, "rb") as f:
                files = {"file": ("test_image.jpg", f, "image/jpeg")}
                response = requests.post(
                    f"{base_url}/detect",
                    files=files,
                    timeout=60
                )
            
            total_time = time.perf_counter() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 檔案上傳檢測成功!")
                
                # 驗證回應格式
                assert 'success' in result, "回應缺少 success 欄位"
                assert result.get('success'), f"檢測失敗: {result.get('message', '未知錯誤')}"
                
                data = result.get('data', result)
                detections = data.get('detections', [])
                assert isinstance(detections, list), "detections 應該是列表"
                
                display_detection_results(result, total_time)
                test_results['file_upload'] = True
                
                # 儲存標註圖片
                annotated_image = data.get('annotated_image', data.get('annotated_image_base64', ''))
                if annotated_image:
                    # 移除 data:image/jpeg;base64, 前綴
                    if annotated_image.startswith('data:image/'):
                        annotated_image = annotated_image.split(',', 1)[-1]
                    save_annotated_image(annotated_image, "result_file_upload.jpg")
                    
            else:
                print(f"   ❌ 檔案上傳檢測失敗: {response.status_code}")
                print(f"   📄 錯誤訊息: {response.text}")
                assert False, f"檔案上傳檢測失敗，狀態碼: {response.status_code}"
                
        except Exception as e:
            print(f"   ❌ 檔案上傳測試失敗: {e}")
            assert False, f"檔案上傳測試異常: {e}"
    else:
        print(f"   ⚠️ 測試圖片不存在: {test_image_path}")
        assert False, f"測試圖片不存在: {test_image_path}"
    
    # 4. 測試 URL 檢測
    print("\n4️⃣ 測試 URL 檢測...")
    
    # 使用測試圖片 URL
    test_image_url = "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg"
    
    try:
        start_time = time.perf_counter()
        
        # 使用 Form 參數格式 (修正後的格式)
        data = {"image_url": test_image_url}
        response = requests.post(
            f"{base_url}/detect",
            data=data,
            timeout=60
        )
        
        total_time = time.perf_counter() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ URL 檢測成功!")
            
            # 驗證回應格式
            assert 'success' in result, "回應缺少 success 欄位"
            assert result.get('success'), f"檢測失敗: {result.get('message', '未知錯誤')}"
            
            data = result.get('data', result)
            detections = data.get('detections', [])
            assert isinstance(detections, list), "detections 應該是列表"
            
            display_detection_results(result, total_time)
            test_results['url_detection'] = True
            
            # 儲存標註圖片
            annotated_image = data.get('annotated_image', data.get('annotated_image_base64', ''))
            if annotated_image:
                # 移除 data:image/jpeg;base64, 前綴
                if annotated_image.startswith('data:image/'):
                    annotated_image = annotated_image.split(',', 1)[-1]
                save_annotated_image(annotated_image, "result_url_detection.jpg")
                
        else:
            print(f"   ❌ URL 檢測失敗: {response.status_code}")
            print(f"   📄 錯誤訊息: {response.text}")
            assert False, f"URL 檢測失敗，狀態碼: {response.status_code}"
            
    except Exception as e:
        print(f"   ❌ URL 檢測測試失敗: {e}")
        assert False, f"URL 檢測異常: {e}"
    
    # 5. 測試錯誤處理
    print("\n5️⃣ 測試錯誤處理...")
    try:
        error_handling_passed = test_error_cases(base_url)
        if error_handling_passed:
            test_results['error_handling'] = True
        else:
            assert False, "錯誤處理測試未通過"
    except Exception as e:
        print(f"   ❌ 錯誤處理測試異常: {e}")
        assert False, f"錯誤處理測試異常: {e}"
    
    print(f"\n✅ API 測試完成")
    
    # 驗證所有關鍵測試都通過
    failed_tests = [test for test, passed in test_results.items() if not passed]
    if failed_tests:
        assert False, f"以下測試未通過: {failed_tests}"
    
    print("🎉 所有測試項目都通過!")

def display_detection_results(result, total_time):
    """顯示檢測結果"""
    # 處理 API 回應格式：result.data.detections
    data = result.get('data', result)  # 兼容新舊格式
    detections = data.get('detections', [])
    
    print(f"   📊 檢測到 {len(detections)} 個藥丸")
    if 'inference_time_ms' in data:
        print(f"   ⚡ 推理時間: {data['inference_time_ms']}ms")
    print(f"   🌐 總請求時間: {total_time*1000:.1f}ms")
    
    # 顯示檢測結果詳情
    for i, detection in enumerate(detections, 1):
        pill_name = detection.get('class_name', detection.get('pill_name', 'Unknown'))
        confidence = detection.get('confidence', 0)
        print(f"      🔍 藥丸 {i}: {pill_name} "
              f"(信心度: {confidence:.3f})")

def test_error_cases(base_url):
    """測試錯誤處理情況"""
    
    invalid_url_test_passed = False
    empty_request_test_passed = False
    
    # 測試無效 URL
    print("   📋 測試無效 URL...")
    try:
        data = {"image_url": "https://invalid-url-that-does-not-exist.com/image.jpg"}
        response = requests.post(f"{base_url}/detect", data=data, timeout=30)
        if response.status_code != 200:
            print(f"   ✅ 無效 URL 錯誤處理正常: {response.status_code}")
            invalid_url_test_passed = True
        else:
            print(f"   ⚠️ 無效 URL 未正確處理")
    except:
        print(f"   ✅ 無效 URL 請求超時，錯誤處理正常")
        invalid_url_test_passed = True
    
    # 測試空請求
    print("   📋 測試空請求...")
    try:
        response = requests.post(f"{base_url}/detect", timeout=10)
        if response.status_code in [400, 422]:  # FastAPI 驗證錯誤或 Bad Request
            print(f"   ✅ 空請求錯誤處理正常: {response.status_code}")
            empty_request_test_passed = True
        else:
            print(f"   ⚠️ 空請求處理異常: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 空請求測試失敗: {e}")
    
    return invalid_url_test_passed and empty_request_test_passed

def save_annotated_image(base64_string, filename):
    """儲存 base64 標註圖片"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # 確保 tests 目錄存在
        output_path = f"tests/{filename}"
        image.save(output_path)
        print(f"   💾 標註圖片已儲存: {output_path}")
    except Exception as e:
        print(f"   ⚠️ 儲存圖片失敗: {e}")

def test_web_interface():
    """測試網頁介面可用性"""
    base_url = "http://localhost:8000"
    
    print("\n🌐 測試網頁介面...")
    try:
        response = requests.get(f"{base_url}/test", timeout=10)
        if response.status_code == 200:
            print("   ✅ 測試介面可存取")
            print(f"   🔗 請訪問: {base_url}/test")
        else:
            print(f"   ❌ 測試介面無法存取: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 測試介面檢查失敗: {e}")

if __name__ == "__main__":
    test_api_comprehensive()
    test_web_interface()