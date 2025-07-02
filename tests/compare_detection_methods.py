#!/usr/bin/env python3
"""
比較純數學方案和Pillow方案在實際圖片上的檢測結果

純數學方案：按照 pure_math_spec.md 實現，只使用 numpy 數學運算
Pillow方案：當前主流程實現，resize → to_tensor → normalize
"""
import numpy as np
import sys
import os
import asyncio
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def pure_math_preprocess_image(image_array):
    """
    純數學方案的前處理 - 按照 pure_math_spec.md
    流程：to_tensor → normalize → resize（全純數學實現）
    """
    # 步驟1: to_tensor - HWC → CHW，轉 float32，/255
    tensor_like = image_array.transpose((2, 0, 1)).astype(np.float32) / 255.0
    
    # 步驟2: normalize - ImageNet 標準化
    means = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
    stds = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
    normalized = (tensor_like - means) / stds
    
    # 步驟3: resize - 純數學雙線性插值實現
    target_h, target_w = 560, 560
    resized_chw = pure_numpy_resize(normalized, (target_h, target_w))
    
    # 步驟4: add batch dimension
    batched = np.expand_dims(resized_chw, axis=0)
    
    return batched

def bilinear_interpolate(image, x, y):
    """雙線性插值算法 - 純數學實現"""
    C, H, W = image.shape
    
    # 獲取四個鄰近像素的整數座標
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, W - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, H - 1)
    
    # 計算插值權重
    wx = x - x0
    wy = y - y0
    
    # 邊界檢查
    x0 = max(0, x0)
    y0 = max(0, y0)
    
    # 對每個通道執行雙線性插值
    result = np.zeros(C)
    for c in range(C):
        result[c] = (1 - wx) * (1 - wy) * image[c, y0, x0] + \
                    wx * (1 - wy) * image[c, y0, x1] + \
                    (1 - wx) * wy * image[c, y1, x0] + \
                    wx * wy * image[c, y1, x1]
    
    return result

def pure_numpy_resize(image, target_size):
    """純數學 resize 實現 - 按照 pure_math_spec.md"""
    C, H, W = image.shape
    target_h, target_w = target_size
    
    # 計算縮放比例
    scale_y = H / target_h
    scale_x = W / target_w
    
    # 創建輸出陣列
    resized = np.zeros((C, target_h, target_w), dtype=np.float32)
    
    # 對每個輸出像素計算對應的輸入座標並插值
    for i in range(target_h):
        for j in range(target_w):
            # 計算在原圖中的浮點座標
            src_y = (i + 0.5) * scale_y - 0.5
            src_x = (j + 0.5) * scale_x - 0.5
            
            # 邊界處理
            src_y = max(0, min(src_y, H - 1))
            src_x = max(0, min(src_x, W - 1))
            
            # 雙線性插值
            resized[:, i, j] = bilinear_interpolate(image, src_x, src_y)
    
    return resized

def pure_math_postprocess_detections(logits, boxes, original_height, original_width, threshold=0.5, top_k=30):
    """純數學方案的後處理 - 按照 pure_math_spec.md 精確實現"""
    # 移除批次維度
    logits = logits[0]  # (300, num_classes)
    boxes = boxes[0]    # (300, 4)
    
    # 計算信心度（使用 sigmoid）
    scores = 1 / (1 + np.exp(-logits))
    
    # 選擇最高信心度的類別（不排除背景類，因為模型沒有背景類）
    max_scores = np.max(scores, axis=1)  # 所有類別
    max_labels = np.argmax(scores, axis=1)
    
    # Top-K 選擇
    top_indices = np.argsort(max_scores)[-top_k:]
    selected_scores = max_scores[top_indices]
    selected_labels = max_labels[top_indices]
    selected_boxes = boxes[top_indices]
    
    # 座標轉換 cxcywh -> xyxy
    def cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)
    
    xyxy_boxes = cxcywh_to_xyxy(selected_boxes)
    
    # 縮放到原圖尺寸
    scale_factors = np.array([original_width, original_height, original_width, original_height])
    scaled_boxes = xyxy_boxes * scale_factors
    
    # 閾值過濾（在座標轉換後）
    keep_mask = selected_scores > threshold
    if not np.any(keep_mask):
        return {'xyxy': np.array([]).reshape(0, 4), 'confidence': np.array([]), 'class_id': np.array([])}
    
    final_scores = selected_scores[keep_mask]
    final_labels = selected_labels[keep_mask]
    final_boxes = scaled_boxes[keep_mask]
    
    # 座標限制
    final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, original_width)
    final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, original_height)
    
    return {
        'xyxy': final_boxes,
        'confidence': final_scores,
        'class_id': final_labels
    }

class TestPillDetector:
    """測試版檢測器，可以切換後處理方法"""
    def __init__(self):
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.pill_detector import PillDetector
        self.detector = PillDetector()
        
    async def initialize(self):
        await self.detector.initialize()
        
    def is_ready(self):
        return self.detector.is_ready()
        
    def get_classes(self):
        return self.detector.get_classes()
        
    async def detect_with_method(self, image_content, method='elegant'):
        """使用指定方法檢測 - 確保調用主流程實現"""
        from PIL import Image
        from io import BytesIO
        
        # 圖像預處理
        image = Image.open(BytesIO(image_content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        original_size = image.size
        image_array = np.array(image)
        
        if method == 'pure_math':
            # 純數學方案：使用純數學前處理
            input_tensor = pure_math_preprocess_image(image_array)
            if input_tensor is None:
                raise ValueError("純數學前處理失敗")
        else:  # elegant
            # Pillow方案：調用主流程前處理
            input_tensor = self.detector.preprocess_image(image_array)
        
        # 模型推理
        input_name = self.detector.onnx_session.get_inputs()[0].name
        outputs = self.detector.onnx_session.run(None, {input_name: input_tensor})
        
        # 根據方法選擇後處理
        if method == 'pure_math':
            # 純數學方案：使用純數學後處理
            detections = pure_math_postprocess_detections(
                outputs[1], outputs[0], 
                original_size[1], original_size[0], 
                threshold=0.5, top_k=30
            )
        else:  # elegant
            # Pillow方案：調用主流程後處理
            detections = self.detector._postprocess_detections(
                outputs[1], outputs[0], 
                original_size[1], original_size[0], 
                threshold=0.5, top_k=30
            )
        
        # 轉換格式
        results = []
        for i in range(len(detections['xyxy'])):
            x1, y1, x2, y2 = detections['xyxy'][i]
            class_id = int(detections['class_id'][i])
            confidence = float(detections['confidence'][i])
            
            results.append({
                'class_id': class_id,
                'class_name': self.detector.class_names[class_id] if self.detector.class_names and class_id < len(self.detector.class_names) else f'Class_{class_id}',
                'confidence': confidence,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
        
        return {
            'detections': results,
            'total_detections': len(results),
            'method': method
        }
    
    def compare_preprocessing(self, image_array):
        """比較兩種前處理方法的差異"""
        print("🔧 比較前處理方法...")
        print("  🎯 純數學方案: to_tensor → normalize → resize（純數學雙線性插值）")
        print("  ✨ Pillow方案: resize → to_tensor → normalize（PIL 在像素域操作）")
        
        # Pillow方案前處理 (當前主流程使用)
        elegant_tensor = self.detector.preprocess_image(image_array)
        
        # 純數學方案前處理 (pure_math_spec.md 實現)
        pure_math_tensor = pure_math_preprocess_image(image_array)
        
        if pure_math_tensor is None:
            print("  ❌ 純數學方案前處理失敗")
            return elegant_tensor, None, None
        
        # 比較結果
        print(f"  Pillow方案輸出形狀: {elegant_tensor.shape}")
        print(f"  純數學方案輸出形狀: {pure_math_tensor.shape}")
        
        diff = np.abs(elegant_tensor - pure_math_tensor)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  最大差異: {max_diff:.6f}")
        print(f"  平均差異: {mean_diff:.6f}")
        print(f"  數值範圍 (Pillow): [{elegant_tensor.min():.3f}, {elegant_tensor.max():.3f}]")
        print(f"  數值範圍 (純數學): [{pure_math_tensor.min():.3f}, {pure_math_tensor.max():.3f}]")
        
        if max_diff < 1e-4:
            print("  ✅ 前處理結果基本一致！")
        elif max_diff < 1e-2:
            print(f"  ⚠️ 前處理結果有小差異，最大差異: {max_diff:.6f}")
        else:
            print(f"  ❌ 前處理結果有顯著差異，最大差異: {max_diff:.6f}")
            print("  💡 這證明了兩種方案確實使用不同的前處理流程！")
        
        return elegant_tensor, pure_math_tensor, max_diff
    
    def verify_main_pipeline_is_elegant(self):
        """驗證主流程確實使用Pillow方案"""
        print("🔍 驗證主流程實現...")
        
        # 檢查前處理方法
        preprocess_code = self.detector.preprocess_image.__doc__ or ""
        if "resize → to_tensor → normalize" in preprocess_code:
            print("  ✅ 主流程前處理確實是Pillow方案")
        else:
            print("  ❌ 主流程前處理可能不是Pillow方案")
            
        # 檢查後處理方法
        postprocess_code = self.detector._postprocess_detections.__doc__ or ""
        if "優雅方案" in postprocess_code:
            print("  ✅ 主流程後處理確實是Pillow方案")
        else:
            print("  ❌ 主流程後處理可能不是Pillow方案")
            
        print("  💡 主流程調用:")
        print("    - 前處理: self.detector.preprocess_image()")
        print("    - 後處理: self.detector._postprocess_detections()")

async def compare_detection_methods():
    """比較兩種檢測方法在實際圖片上的效果"""
    print("🔬 比較純數學方案 vs Pillow方案 - 五張實際圖片檢測")
    print("=" * 70)
    
    # 初始化檢測器
    detector = TestPillDetector()
    await detector.initialize()
    
    if not detector.is_ready():
        print("❌ 檢測器初始化失敗")
        return
        
    print("✅ 檢測器初始化成功")
    print(f"📋 支援類別: {detector.get_classes()}")
    
    # 驗證主流程確實使用Pillow方案
    print(f"\n🔍 主流程實現驗證")
    print("-"*50)
    detector.verify_main_pipeline_is_elegant()
    
    # 獲取所有測試圖片
    import glob
    image_files = sorted(glob.glob("/workspace/tests/IMG_*.jpg"))
    
    if not image_files:
        print("❌ 沒有找到測試圖片")
        return
        
    print(f"📸 找到 {len(image_files)} 張測試圖片")
    
    # 前處理驗證 - 只需要測試一張圖片
    print(f"\n🔧 前處理方法驗證")
    print("-"*50)
    
    first_image_path = image_files[0]
    with open(first_image_path, 'rb') as f:
        first_image_content = f.read()
    
    # 載入圖片進行前處理比較
    from PIL import Image
    from io import BytesIO
    image = Image.open(BytesIO(first_image_content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image)
    
    # 比較前處理
    elegant_tensor, pure_math_tensor, max_diff = detector.compare_preprocessing(image_array)
    
    print(f"\n🖼️  檢測結果比較")
    print("-"*50)
    
    all_pure_results = []
    all_elegant_results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n🖼️  圖片 {i}: {os.path.basename(image_path)}")
        
        try:
            # 讀取圖片
            with open(image_path, 'rb') as f:
                image_content = f.read()
            
            # 純數學方案檢測
            result_pure = await detector.detect_with_method(image_content, method='pure_math')
            all_pure_results.append(result_pure)
            
            print(f"  🧮 純數學方案: {result_pure['total_detections']} 個檢測")
            if result_pure['detections']:
                for j, det in enumerate(result_pure['detections'], 1):
                    print(f"    {j}. {det['class_name']} (信心度: {det['confidence']:.3f})")
            
            # Pillow方案檢測
            result_elegant = await detector.detect_with_method(image_content, method='elegant')
            all_elegant_results.append(result_elegant)
            
            print(f"  ✨ Pillow方案: {result_elegant['total_detections']} 個檢測")
            if result_elegant['detections']:
                for j, det in enumerate(result_elegant['detections'], 1):
                    print(f"    {j}. {det['class_name']} (信心度: {det['confidence']:.3f})")
            
        except Exception as e:
            print(f"  ❌ 處理失敗: {e}")
    
    # 整體分析
    print(f"\n{'='*70}")
    print("📊 整體結果分析")
    print(f"{'='*70}")
    
    # 統計總檢測數
    total_pure = sum(r['total_detections'] for r in all_pure_results)
    total_elegant = sum(r['total_detections'] for r in all_elegant_results)
    
    print(f"📈 檢測數量統計:")
    print(f"  純數學方案總檢測數: {total_pure}")
    print(f"  Pillow方案總檢測數: {total_elegant}")
    print(f"  差異: {abs(total_pure - total_elegant)}")
    
    # 平均信心度分析
    all_pure_confidences = []
    all_elegant_confidences = []
    
    for result in all_pure_results:
        all_pure_confidences.extend([det['confidence'] for det in result['detections']])
    
    for result in all_elegant_results:
        all_elegant_confidences.extend([det['confidence'] for det in result['detections']])
    
    if all_pure_confidences and all_elegant_confidences:
        pure_avg_conf = np.mean(all_pure_confidences)
        elegant_avg_conf = np.mean(all_elegant_confidences)
        
        print(f"\n🎯 信心度分析:")
        print(f"  純數學方案平均信心度: {pure_avg_conf:.3f}")
        print(f"  Pillow方案平均信心度: {elegant_avg_conf:.3f}")
        print(f"  信心度差異: {abs(pure_avg_conf - elegant_avg_conf):.3f}")
        
        print(f"  純數學方案最高信心度: {max(all_pure_confidences):.3f}")
        print(f"  Pillow方案最高信心度: {max(all_elegant_confidences):.3f}")
    
    # 檢測到的藥丸類別比較
    print(f"\n💊 檢測藥丸類別分析:")
    
    for i, (pure_result, elegant_result) in enumerate(zip(all_pure_results, all_elegant_results), 1):
        pure_classes = set([det['class_name'] for det in pure_result['detections']])
        elegant_classes = set([det['class_name'] for det in elegant_result['detections']])
        
        common_classes = pure_classes.intersection(elegant_classes)
        only_pure = pure_classes - elegant_classes
        only_elegant = elegant_classes - pure_classes
        
        print(f"  圖片 {i}:")
        print(f"    共同檢測到: {list(common_classes) if common_classes else '無'}")
        if only_pure:
            print(f"    僅純數學檢測到: {list(only_pure)}")
        if only_elegant:
            print(f"    僅Pillow方案檢測到: {list(only_elegant)}")
        
        if pure_classes == elegant_classes:
            print(f"    ✅ 兩方案檢測類別完全一致")
        else:
            print(f"    ⚠️ 兩方案檢測類別存在差異")
    
    print(f"\n🎯 總結:")
    print(f"  - 純數學方案總共檢測到 {total_pure} 個藥丸")
    print(f"  - Pillow方案總共檢測到 {total_elegant} 個藥丸")
    if all_pure_confidences and all_elegant_confidences:
        print(f"  - 純數學方案平均信心度: {np.mean(all_pure_confidences):.3f}")
        print(f"  - Pillow方案平均信心度: {np.mean(all_elegant_confidences):.3f}")
    print(f"  - Pillow方案代碼更簡潔，避免同位置多類別問題")

if __name__ == "__main__":
    asyncio.run(compare_detection_methods())
