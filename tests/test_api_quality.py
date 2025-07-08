#!/usr/bin/env python3
"""
測試API品質分析功能
驗證API回應是否包含品質分析欄位
"""
import requests
import json

def test_api_quality_analysis():
    base_url = "http://localhost:8000"
    
    print("🔍 測試API品質分析功能...")
    
    # 測試1: 檔案上傳檢測
    print("\n1️⃣ 測試檔案上傳檢測的品質分析...")
    
    test_image_path = "tests/IMG_1363_JPG.rf.47f388a7df161e710d93964e509026d6.jpg"
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/detect", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ 檔案上傳檢測成功")
            print(f"📊 檢測到 {result['data']['total_detections']} 個藥丸")
            
            # 檢查品質分析
            if 'quality_analysis' in result['data']:
                qa = result['data']['quality_analysis']
                print(f"\n🔍 品質分析結果:")
                print(f"   是否建議重拍: {qa.get('should_retake', 'N/A')}")
                print(f"   原因: {qa.get('reason', 'N/A')}")
                print(f"   訊息: {qa.get('message', 'N/A')}")
                
                if 'quality_score' in qa:
                    print(f"   品質分數: {qa['quality_score']:.2f}")
                
                if 'suggestions' in qa and qa['suggestions']:
                    print(f"   建議:")
                    for suggestion in qa['suggestions']:
                        print(f"     • {suggestion}")
                
                if 'uncertain_items' in qa and qa['uncertain_items']:
                    print(f"   低信心度項目: {qa['uncertain_items']}")
                
                print(f"\n✅ API已成功返回品質分析！")
            else:
                print(f"❌ API回應中沒有找到 quality_analysis 欄位")
                print(f"🔍 回應結構: {list(result['data'].keys())}")
        else:
            print(f"❌ 檔案上傳檢測失敗: {response.status_code}")
            print(f"回應: {response.text}")
            
    except Exception as e:
        print(f"❌ 檔案上傳測試失敗: {e}")
    
    # 測試2: URL檢測
    print("\n2️⃣ 測試URL檢測的品質分析...")
    
    test_url = "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg"
    
    try:
        data = {'image_url': test_url}
        response = requests.post(f"{base_url}/detect", data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ URL檢測成功")
            print(f"📊 檢測到 {result['data']['total_detections']} 個藥丸")
            
            # 檢查品質分析
            if 'quality_analysis' in result['data']:
                qa = result['data']['quality_analysis']
                print(f"\n🔍 品質分析結果:")
                print(f"   是否建議重拍: {qa.get('should_retake', 'N/A')}")
                print(f"   原因: {qa.get('reason', 'N/A')}")
                print(f"   訊息: {qa.get('message', 'N/A')}")
                
                if 'quality_score' in qa:
                    print(f"   品質分數: {qa['quality_score']:.2f}")
                
                print(f"\n✅ URL檢測也成功返回品質分析！")
            else:
                print(f"❌ URL檢測回應中沒有找到 quality_analysis 欄位")
        else:
            print(f"❌ URL檢測失敗: {response.status_code}")
            print(f"回應: {response.text}")
            
    except Exception as e:
        print(f"❌ URL檢測測試失敗: {e}")
    
    # 測試3: 檢查API回應格式
    print("\n3️⃣ 檢查完整API回應格式...")
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/detect", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📋 API回應結構:")
            print(f"   top level keys: {list(result.keys())}")
            print(f"   data keys: {list(result['data'].keys())}")
            
            # 檢查是否向後相容
            required_keys = ['detections', 'annotated_image', 'total_detections']
            missing_keys = [key for key in required_keys if key not in result['data']]
            
            if not missing_keys:
                print(f"✅ API向後相容性良好，保留了所有必要欄位")
            else:
                print(f"⚠️  缺少必要欄位: {missing_keys}")
            
            # 檢查新增的品質分析欄位
            if 'quality_analysis' in result['data']:
                qa_keys = list(result['data']['quality_analysis'].keys())
                print(f"✅ 新增了 quality_analysis 欄位，包含: {qa_keys}")
            else:
                print(f"❌ 未找到 quality_analysis 欄位")
                
    except Exception as e:
        print(f"❌ API格式檢查失敗: {e}")

if __name__ == "__main__":
    test_api_quality_analysis()