"""
藥丸檢測核心模組 - RF-DETR ONNX推理引擎

核心特色：
- 優雅預處理流程：resize → to_tensor → normalize（在原始像素域操作，避免精度損失）
- 完全匹配RF-DETR官方predict方法結果，但使用ONNX推理
- 僅依賴 numpy + Pillow，移除PyTorch依賴
- 職責分離：專注檢測，標註由detection_service處理

技術規範參考：
- docs/01_TECHNICAL_JOURNEY_COMPACT.md（演進總覽）
- legacy/elegant_solution_spec.md（優雅方案詳細說明）
- legacy/rfdetr_original_spec.md（原始RF-DETR實現規範）
"""
import os
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import *
from .utils.coordinate_utils import CoordinateUtils

logger = logging.getLogger(__name__)

class PillDetector:
    """
    藥丸檢測器 - RF-DETR ONNX推理核心
    
    實現RF-DETR官方predict方法的ONNX版本，保持完全一致的結果：
    
    架構設計：
    - 預處理：優雅的 resize → to_tensor → normalize 流程
    - 推理：ONNX Runtime高效推理，替代PyTorch
    - 後處理：每位置最高分類 → 閾值過濾 → Top-K選擇（符合物理直觀）
    - 座標轉換：cxcywh → xyxy，並縮放到處理後圖像尺寸
    
    技術特點：
    - 在原始像素域（uint8 [0,255]）進行resize，避免精度損失
    - 支援長寬比保持的resize + padding到560x560
    - 僅依賴 numpy + Pillow + ONNX Runtime，無OpenCV
    - 職責單一：僅負責檢測，標註交由detection_service
    """
    
    def __init__(self):
        self.onnx_session = None
        self.class_names = None
        # 移除ImageAnnotator實例，由detection_service統一管理
        
    async def initialize(self):
        """初始化模型和類別名稱"""
        await self._load_model()
        await self._load_class_names()
        
    async def _load_model(self):
        """載入 ONNX 模型"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"ONNX 模型檔案不存在: {MODEL_PATH}")
                
            self.onnx_session = ort.InferenceSession(
                MODEL_PATH,
                providers=['CPUExecutionProvider']
            )
            logger.info("✅ ONNX 模型載入成功")
            logger.info(f"🔍 模型輸入: {[inp.name for inp in self.onnx_session.get_inputs()]}")
            logger.info(f"📤 模型輸出: {[out.name for out in self.onnx_session.get_outputs()]}")
            
        except Exception as e:
            logger.error(f"❌ 載入 ONNX 模型失敗: {e}")
            raise
            
    async def _load_class_names(self):
        """載入類別名稱"""
        try:
            if not os.path.exists(COCO_ANNOTATIONS_PATH):
                raise FileNotFoundError(f"COCO 標註檔案不存在: {COCO_ANNOTATIONS_PATH}")
                
            with open(COCO_ANNOTATIONS_PATH, "r", encoding='utf-8') as f:
                coco_data = json.load(f)
            
            categories = sorted(coco_data['categories'], key=lambda x: x['id'])
            self.class_names = [cat['name'] for cat in categories]
            
            logger.info(f"✅ 成功載入 {len(self.class_names)} 個類別")
            logger.info(f"📋 類別: {self.class_names}")
            
        except Exception as e:
            logger.error(f"❌ 載入類別名稱失敗: {e}")
            raise
            
    def preprocess_image(self, image_array: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
        """
        優雅預處理實現 - 完全匹配RF-DETR官方流程
        
        操作順序（關鍵）：resize → to_tensor → normalize
        
        為什麼這個順序重要：
        - RF-DETR官方: PIL → to_tensor → normalize → resize
        - 我們的優雅方案: PIL → resize → to_tensor → normalize
        - 在原始像素域（uint8 [0,255]）resize避免標準化數據的精度損失
        
        具體流程：
        1. numpy array → PIL Image
        2. 長寬比保持resize + 黑色padding到560x560
        3. PIL → tensor (CHW format, [0,1] range)
        4. ImageNet標準化: (pixel - mean) / std
        5. 添加batch維度: (1, C, H, W)
        
        Args:
            image_array: 輸入圖像numpy陣列 (H,W,C) uint8 [0,255]
            
        Returns:
            (input_tensor, processed_image): 
            - input_tensor: ONNX模型輸入 (1,C,H,W) float32，ImageNet標準化
            - processed_image: 處理後的PIL圖像 (560,560) RGB，用於後續座標計算
            
        參考：legacy/elegant_solution_spec.md
        """
        try:
            # === 步驟1: 將 numpy array 轉為 PIL Image ===
            pil_image = Image.fromarray(image_array.astype(np.uint8))
            
            # === 步驟2: 保持長寬比resize + padding到560x560 ===
            target_h, target_w = INPUT_SIZE[1], INPUT_SIZE[0]  # INPUT_SIZE 是 (width, height)
            
            # 計算保持長寬比的縮放比例
            original_w, original_h = pil_image.size
            scale = min(target_w / original_w, target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            # 先resize到合適大小（保持長寬比）
            resized_pil = pil_image.resize((new_w, new_h), Image.BILINEAR)
            
            # 創建560x560的黑色背景圖片
            padded_pil = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            
            # 計算居中位置
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            
            # 將resize後的圖片貼到中心
            padded_pil.paste(resized_pil, (paste_x, paste_y))
            resized_pil = padded_pil
            
            # === 步驟3: to_tensor ===
            # PIL Image (H,W,C) uint8 [0,255] → numpy (C,H,W) float32 [0,1]
            np_img = np.array(resized_pil).astype(np.float32) / 255.0  # [0,1]
            tensor_img = np.transpose(np_img, (2, 0, 1))  # HWC → CHW
            
            # === 步驟4: normalize ===
            # ImageNet 標準化: normalized = (image - mean) / std
            means = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            stds = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
            normalized_img = (tensor_img - means) / stds
            
            # === 步驟5: 添加 batch 維度 ===
            # (C, H, W) → (1, C, H, W)
            batched = np.expand_dims(normalized_img, axis=0)
            
            return batched, resized_pil
            
        except Exception as e:
            logger.error(f"❌ 圖像預處理失敗: {e}")
            raise
        
    def postprocess_results(self, outputs) -> List[Dict]:
        """
        ONNX模型輸出後處理 - 優雅方案實現
        
        對比RF-DETR官方實現：
        - 官方: 全域Top-K搜索（從300×6=1800個值中選Top-100）
        - 優雅方案: 每位置最高分類 → 閾值過濾 → Top-K選擇
        
        優雅方案技術優勢：
        1. 避免同位置多檢測（符合物理直觀，一個位置只能有一個物體）
        2. 算法邏輯更清晰直觀，便於理解和維護
        3. 性能更優，減少不必要的計算開銷
        4. 結果更符合實際應用場景需求
        
        處理流程（按技術規范）：
        1. 提取模型輸出: pred_boxes (1,300,4), pred_logits (1,300,num_classes)
        2. Sigmoid激活: logits → confidence scores [0,1]
        3. 每位置最高分類: 300個位置各選最佳類別（優雅方案核心）
        4. 閾值過濾: 保留 confidence > CONFIDENCE_THRESHOLD 的檢測
        5. Top-K選擇: 從過濾結果中選擇前TOP_K個最佳檢測
        6. 座標轉換: cxcywh → xyxy，縮放到處理後圖像尺寸(560,560)
        
        Args:
            outputs: ONNX Runtime模型輸出列表 [pred_boxes, pred_logits]
                - pred_boxes: (1,300,4) 邊界框預測，cxcywh格式，歸一化[0,1]
                - pred_logits: (1,300,num_classes) 類別預測logits
            處理後圖像尺寸由INPUT_SIZE配置決定，無需外部傳入
            
        Returns:
            List[Dict]: 檢測結果列表，每項包含:
                - class_id: int, 類別索引(0-based)
                - class_name: str, 類別名稱（從COCO標註文件獲取）
                - confidence: float, 信心度分數[0,1]
                - bbox: List[int], 邊界框[x1,y1,x2,y2]，絕對像素座標
            
        技術參考：
        - legacy/rfdetr_original_spec.md: RF-DETR官方實現細節
        - legacy/elegant_solution_spec.md: 優雅方案設計理念
        - docs/01_TECHNICAL_JOURNEY_COMPACT.md: 後處理算法演進
        """
        try:
            # === 步驟1: 提取模型輸出 ===
            # 模型輸出格式：[boxes, logits]
            pred_boxes = outputs[0]   # (1, 300, 4) 邊界框預測
            pred_logits = outputs[1]  # (1, 300, num_classes) 類別預測
            
            # 使用優雅方案的後處理函數（圖像尺寸固定為560×560）
            detections = self._postprocess_detections(
                pred_logits, pred_boxes, 
                threshold=CONFIDENCE_THRESHOLD, 
                top_k=TOP_K
            )
            
            # 轉換為 API 輸出格式
            results = []
            for i in range(len(detections['xyxy'])):
                x1, y1, x2, y2 = detections['xyxy'][i]
                class_id = int(detections['class_id'][i])
                confidence = float(detections['confidence'][i])
                
                results.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if self.class_names and class_id < len(self.class_names) else f'Class_{class_id}',
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                    
            return results
                
        except Exception as e:
            logger.error(f"❌ 後處理失敗: {e}")
            return []
            
    def _postprocess_detections(self, logits, boxes, threshold=0.5, top_k=100):
        """
        優雅方案後處理核心實現 - 每位置最高分類策略
        
        與RF-DETR官方後處理器的關鍵差異：
        - 官方: 全域Top-K (從300×6=1800個分數中選Top-100)
        - 優雅: 每位置最佳 → 閾值過濾 → Top-K (避免同位置多檢測)
        
        算法流程（基於技術規範）:
        1. Sigmoid激活: 將logits轉為[0,1]概率分數
        2. 每位置最高分類: 300個查詢位置各選最佳類別
        3. 閾值過濾: 保留信心度 > threshold 的檢測
        4. 信心度排序: 按分數降序排列
        5. Top-K選擇: 選擇前K個最佳檢測（默認100個）
        6. 座標轉換: cxcywh → xyxy，縮放到處理圖像尺寸
        
        Args:
            logits: (1, 300, num_classes) ONNX模型類別預測logits
            boxes: (1, 300, 4) ONNX模型框預測，cxcywh格式，歸一化[0,1]
            threshold: float, 信心度閾值，默認0.5
            top_k: int, 最大檢測數量，默認100
            
        Note:
            處理後圖像尺寸由INPUT_SIZE配置決定，座標縮放使用配置值
            
        Returns:
            Dict: 包含以下鍵值的檢測結果字典
                - 'xyxy': (N,4) numpy array, 絕對像素座標邊界框
                - 'confidence': (N,) numpy array, 信心度分數[0,1]
                - 'class_id': (N,) numpy array, 類別索引(0-based)
                
        技術特點:
        - 物理直觀: 一個位置只檢測一個物體（符合現實）
        - 計算高效: 避免不必要的全域搜索
        - 邏輯清晰: 步驟明確，易於理解和除錯
        """
        # 移除批次維度
        logits = logits[0]  # (300, num_classes)
        boxes = boxes[0]    # (300, 4)
        
        # 計算信心度（使用 sigmoid）
        scores = 1 / (1 + np.exp(-logits))  # sigmoid 函數
        
        # 步驟1：每個位置取最高分類（優雅方案核心）
        max_scores = np.max(scores, axis=1)  # (300,) 每個位置的最高分數
        max_classes = np.argmax(scores, axis=1)  # (300,) 每個位置的最佳類別
        
        # 步驟2：閾值過濾
        keep_mask = max_scores > threshold
        if not np.any(keep_mask):
            return {'xyxy': np.array([]).reshape(0, 4), 'confidence': np.array([]), 'class_id': np.array([])}
            
        final_scores = max_scores[keep_mask]
        final_classes = max_classes[keep_mask]
        final_boxes = boxes[keep_mask]
        
        # 步驟3：Top-K 選擇（在過濾後的結果中選擇）
        if len(final_scores) > top_k:
            top_indices = np.argsort(final_scores)[-top_k:]
            final_scores = final_scores[top_indices]
            final_classes = final_classes[top_indices]
            final_boxes = final_boxes[top_indices]
        
        # 座標轉換: cxcywh → xyxy，並縮放到配置指定的絕對座標
        xyxy_boxes = CoordinateUtils.cxcywh_to_xyxy(final_boxes)
        target_size = INPUT_SIZE[0]  # 使用配置中的尺寸（width = height）
        scaled_boxes = xyxy_boxes * target_size
        
        return {
            'xyxy': scaled_boxes,
            'confidence': final_scores,
            'class_id': final_classes
        }
    
        
    def get_classes(self) -> List[str]:
        """
        取得所有支援的藥丸類別名稱
        
        從COCO標註文件載入的類別清單，用於API端點 /classes
        
        Returns:
            List[str]: 藥丸類別名稱列表，依照COCO類別ID排序
                      例如: ['Amoxicillin', 'Diovan 160mg', 'Lansoprazole', ...]
        """
        return self.class_names or []
        
    def is_ready(self) -> bool:
        """
        檢查檢測器是否已完成初始化
        
        驗證ONNX模型和類別名稱是否都已成功載入，
        確保檢測器可以正常執行推理任務。
        
        Returns:
            bool: True表示檢測器已就緒，False表示仍在初始化中
        """
        return self.onnx_session is not None and self.class_names is not None
    
