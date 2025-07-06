#!/usr/bin/env python3
"""
調試檢測結果腳本
分析為什麼file upload只檢測到10個目標，以及URL detection的Diovan重複問題
"""
import asyncio
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, '/workspace')

from app.detection_service import DetectionService
from app.pill_detector import PillDetector
from app.config import TOP_K, CONFIDENCE_THRESHOLD

async def debug_detection():
    print("🔍 開始調試檢測結果...")
    print(f"📊 配置: TOP_K={TOP_K}, CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}")
    
    # 初始化服務
    service = DetectionService()
    await service.initialize()
    
    # 測試本地圖片 (file upload 場景)
    print("\n" + "="*60)
    print("📁 測試本地圖片檢測 (File Upload)")
    print("="*60)
    
    test_image_path = "tests/IMG_1363_JPG.rf.47f388a7df161e710d93964e509026d6.jpg"
    
    try:
        with open(test_image_path, 'rb') as f:
            image_content = f.read()
        
        # 載入圖片用於詳細分析
        image = Image.open(test_image_path)
        image_array = np.array(image.convert('RGB'))
        
        print(f"🖼️  圖片尺寸: {image.size}")
        print(f"📊 圖片陣列形狀: {image_array.shape}")
        
        # 執行檢測，但深入後處理過程
        input_tensor, processed_image = service.detector.preprocess_image(image_array)
        print(f"⚙️  預處理後張量形狀: {input_tensor.shape}")
        
        # 模型推理
        outputs = service.detector.onnx_session.run(None, {service._onnx_input_name: input_tensor})
        print(f"🎯 模型輸出形狀:")
        for i, output in enumerate(outputs):
            print(f"   Output {i}: {output.shape}")
        
        # 詳細分析後處理過程
        pred_boxes = outputs[0]  # (1, 30, 4)
        pred_logits = outputs[1]  # (1, 30, num_classes)
        
        print(f"\n🔍 詳細後處理分析:")
        print(f"📦 預測框形狀: {pred_boxes.shape}")
        print(f"📊 預測邏輯形狀: {pred_logits.shape}")
        
        # 移除批次維度
        logits = pred_logits[0]  # (30, num_classes)
        boxes = pred_boxes[0]    # (30, 4)
        
        # 計算信心度
        scores = 1 / (1 + np.exp(-logits))  # sigmoid
        max_scores = np.max(scores, axis=1)  # 每個位置的最高分數
        max_classes = np.argmax(scores, axis=1)  # 每個位置的最佳類別
        
        print(f"🎯 所有30個候選位置的最高信心度:")
        for i in range(len(max_scores)):
            if max_scores[i] > 0.5:  # 只顯示信心度 > 0.5 的
                print(f"   位置 {i:2d}: 信心度 {max_scores[i]:.3f}, 類別 {max_classes[i]:2d}")
        
        # 閾值過濾
        keep_mask = max_scores > CONFIDENCE_THRESHOLD
        num_above_threshold = np.sum(keep_mask)
        print(f"\n📊 閾值過濾結果:")
        print(f"   信心度 > {CONFIDENCE_THRESHOLD} 的目標數量: {num_above_threshold}")
        
        if num_above_threshold > 0:
            filtered_scores = max_scores[keep_mask]
            filtered_classes = max_classes[keep_mask]
            filtered_boxes = boxes[keep_mask]
            
            print(f"   過濾後的信心度範圍: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}")
            
            # Top-K 選擇
            if len(filtered_scores) > TOP_K:
                print(f"🔢 需要 Top-K 選擇: {len(filtered_scores)} → {TOP_K}")
                top_indices = np.argsort(filtered_scores)[-TOP_K:]
                final_count = TOP_K
            else:
                print(f"🔢 無需 Top-K 選擇: {len(filtered_scores)} ≤ {TOP_K}")
                final_count = len(filtered_scores)
            
            print(f"✅ 最終檢測數量: {final_count}")
        
        # 執行完整檢測
        result = await service.detect_from_file(image_content, "test.jpg")
        print(f"\n🎉 完整檢測結果: {result['total_detections']} 個目標")
        
    except Exception as e:
        print(f"❌ 本地圖片檢測失敗: {e}")
    
    # 測試URL檢測 (檢查Diovan重複問題)
    print("\n" + "="*60)
    print("🌐 測試URL檢測 (Diovan重複問題分析)")
    print("="*60)
    
    test_url = "https://i.postimg.cc/fRrjZ0DK/IMG-1237-JPG-rf-65888afb7f3a5acce6b2cfa2106a9040.jpg"
    
    try:
        result = await service.detect_from_url(test_url)
        
        print(f"🎯 URL檢測結果: {result['total_detections']} 個目標")
        print(f"📋 檢測詳細:")
        
        # 分析Diovan重複問題
        diovan_detections = []
        for i, detection in enumerate(result['detections']):
            print(f"   {i+1}. {detection['class_name']} - 信心度: {detection['confidence']:.3f}")
            print(f"      位置: {detection['bbox']}")
            
            if 'Diovan' in detection['class_name_en']:
                diovan_detections.append({
                    'index': i,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence']
                })
        
        if len(diovan_detections) > 1:
            print(f"\n⚠️  發現 {len(diovan_detections)} 個 Diovan 檢測:")
            for i, det in enumerate(diovan_detections):
                print(f"   Diovan {i+1}: 位置 {det['bbox']}, 信心度 {det['confidence']:.3f}")
            
            # 計算兩個框的重疊
            if len(diovan_detections) == 2:
                bbox1 = diovan_detections[0]['bbox']
                bbox2 = diovan_detections[1]['bbox']
                
                # 計算IoU
                x1_inter = max(bbox1[0], bbox2[0])
                y1_inter = max(bbox1[1], bbox2[1])
                x2_inter = min(bbox1[2], bbox2[2])
                y2_inter = min(bbox1[3], bbox2[3])
                
                if x1_inter < x2_inter and y1_inter < y2_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    union_area = area1 + area2 - inter_area
                    iou = inter_area / union_area
                    
                    print(f"   📐 兩個 Diovan 框的 IoU: {iou:.3f}")
                    if iou > 0.5:
                        print(f"   ⚠️  IoU > 0.5，可能是同位置重複檢測！")
                    else:
                        print(f"   ✅ IoU ≤ 0.5，屬於不同位置的正常檢測")
                else:
                    print(f"   ✅ 兩個框沒有重疊，屬於不同位置")
        
    except Exception as e:
        print(f"❌ URL檢測失敗: {e}")

if __name__ == "__main__":
    asyncio.run(debug_detection())