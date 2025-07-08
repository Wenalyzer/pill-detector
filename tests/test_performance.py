#!/usr/bin/env python3
import sys
import os
import random
import time
from contextlib import contextmanager
from PIL import Image
import asyncio

# 導入我們的檢測模組
from app.detection_service import DetectionService
from app.pill_detector import PillDetector
from app.image_annotator import ImageAnnotator
from app.config import CONFIDENCE_THRESHOLD

@contextmanager
def timer(description):
    """Context manager for precise timing"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {end - start:.2f}s")

async def predict_and_annotate(image_path, threshold=None):
    """使用當前ONNX架構進行預測和標註"""
    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD
    
    with timer("Total execution time"):
        
        # 初始化檢測服務
        with timer("Model loading"):
            detection_service = DetectionService()
            await detection_service.initialize()
        
        # 載入圖像
        with timer("Image loading"):
            image = Image.open(image_path)
        
        # 執行檢測
        with timer("Prediction"):
            results = await detection_service._detect_and_annotate(image)
        
        # 標註圖像
        with timer("Annotation"):
            # 檢測結果已經包含標註後的圖像
            pass
        
        # 儲存標註圖像
        output_path = "annotated_" + os.path.basename(image_path)
        
        # 從base64解碼並儲存
        import base64
        
        if results.get("annotated_image"):
            # 移除base64前綴
            base64_data = results["annotated_image"].split(',')[1] if ',' in results["annotated_image"] else results["annotated_image"]
            image_data = base64.b64decode(base64_data)
            
            with open(output_path, 'wb') as f:
                f.write(image_data)
    
    return results, output_path

async def main():
    # 預設行為：隨機圖像使用預設閾值
    if len(sys.argv) == 1:
        # 從測試數據中選擇隨機圖像
        test_images = [f for f in os.listdir("tests") if f.endswith(".jpg")]
        image_path = os.path.join("tests", random.choice(test_images))
        threshold = None  # 使用預設閾值
    elif len(sys.argv) == 2:
        if sys.argv[1] == "random":
            test_images = [f for f in os.listdir("tests") if f.endswith(".jpg")]
            image_path = os.path.join("tests", random.choice(test_images))
            threshold = None
        else:
            image_path = sys.argv[1]
            threshold = None
    else:
        if sys.argv[1] == "random":
            test_images = [f for f in os.listdir("tests") if f.endswith(".jpg")]
            image_path = os.path.join("tests", random.choice(test_images))
            threshold = float(sys.argv[2])
        else:
            image_path = sys.argv[1]
            threshold = float(sys.argv[2])
    
    print(f"Processing image: {image_path}")
    print(f"Threshold: {threshold if threshold else CONFIDENCE_THRESHOLD}")
    print("-" * 50)
    
    results, output_path = await predict_and_annotate(image_path, threshold)
    
    print("-" * 50)
    print(f"Input image: {image_path}")
    print(f"Found {len(results.get('detections', []))} detections")
    print(f"Annotated image saved to: {output_path}")
    
    if results.get('detections'):
        print("Detected objects:")
        for i, detection in enumerate(results['detections']):
            print(f"  {i+1}. Class: {detection['class_name']}, Confidence: {detection['confidence']:.3f}")
    else:
        print("No objects detected.")

if __name__ == "__main__":
    asyncio.run(main())