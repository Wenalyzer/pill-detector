"""
藥丸檢測核心模組 - RF-DETR ONNX推理引擎

主要功能：
- 預處理流程：resize → to_tensor → normalize（在原始像素域操作，避免精度損失）
- RF-DETR ONNX 模型推理
- 依賴 numpy + Pillow，移除PyTorch依賴
- 職責分離：專注檢測，標註由detection_service處理

技術規範參考：
- docs/01_TECHNICAL_JOURNEY_COMPACT.md（演進總覽）
- legacy/elegant_solution_spec.md（Pillow 實現詳細說明）
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
    
    實現RF-DETR predict方法的ONNX版本：
    
    架構設計：
    - 預處理：resize → to_tensor → normalize 流程
    - 推理：ONNX Runtime 推理，替代PyTorch
    - 後處理：每位置最高分類 → 閾值過濾 → Top-K選擇
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
                logger.error(f"ONNX 模型檔案不存在: {MODEL_PATH}")
                raise FileNotFoundError("ONNX 模型檔案不存在")
            
            # 配置 ONNX Runtime 性能優化設定
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = INTER_OP_NUM_THREADS
            sess_options.intra_op_num_threads = INTRA_OP_NUM_THREADS
            sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, GRAPH_OPTIMIZATION_LEVEL)
            sess_options.execution_mode = getattr(ort.ExecutionMode, EXECUTION_MODE)
            sess_options.enable_mem_pattern = ENABLE_MEM_PATTERN
            sess_options.enable_cpu_mem_arena = ENABLE_CPU_MEM_ARENA
            
            # 可選性能分析（調試用）
            if ENABLE_PROFILING:
                sess_options.enable_profiling = True
                logger.info("🔍 ONNX 性能分析已啟用")
                
            self.onnx_session = ort.InferenceSession(
                MODEL_PATH,
                sess_options=sess_options,
                providers=ONNX_PROVIDERS
            )
            logger.info("✅ ONNX 模型載入成功")
            logger.info(f"🔍 模型輸入: {[inp.name for inp in self.onnx_session.get_inputs()]}")
            logger.info(f"📤 模型輸出: {[out.name for out in self.onnx_session.get_outputs()]}")
            
        except Exception as e:
            logger.error(f"❌ 載入 ONNX 模型失敗: {str(e)}")
            raise Exception("模型載入失敗")
            
    async def _load_class_names(self):
        """載入類別名稱"""
        try:
            if not os.path.exists(COCO_ANNOTATIONS_PATH):
                logger.error(f"COCO 標註檔案不存在: {COCO_ANNOTATIONS_PATH}")
                raise FileNotFoundError("類別定義檔案不存在")
                
            with open(COCO_ANNOTATIONS_PATH, "r", encoding='utf-8') as f:
                coco_data = json.load(f)
            
            categories = sorted(coco_data['categories'], key=lambda x: x['id'])
            self.class_names = [cat['name'] for cat in categories]
            
            logger.info(f"✅ 成功載入 {len(self.class_names)} 個類別")
            logger.info(f"📋 類別: {self.class_names}")
            
        except Exception as e:
            logger.error(f"❌ 載入類別名稱失敗: {str(e)}")
            raise Exception("類別定義載入失敗")
            
    def preprocess_image(self, image_array: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
        """
        預處理實現 - 參考RF-DETR官方流程
        
        操作順序：resize → to_tensor → normalize
        
        順序調整原因：
        - RF-DETR官方: PIL → to_tensor → normalize → resize
        - 修改後方案: PIL → resize → to_tensor → normalize
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
            resized_pil = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
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
        ONNX模型輸出後處理
        
        Args:
            outputs: ONNX Runtime模型輸出列表 [pred_boxes, pred_logits]
            
        Returns:
            List[Dict]: 檢測結果列表，包含 class_id, class_name, confidence, bbox
        """
        try:
            # === 步驟1: 提取模型輸出 ===
            # 模型輸出格式：[boxes, logits]
            pred_boxes = outputs[0]   # (1, 300, 4) 邊界框預測
            pred_logits = outputs[1]  # (1, 300, num_classes) 類別預測
            
            # 使用修改方案的後處理函數（圖像尺寸固定為560×560）
            detections = self._postprocess_detections(
                pred_logits, pred_boxes, 
                threshold=CONFIDENCE_THRESHOLD, 
                top_k=TOP_K
            )
            
            # 分析相似外觀檢測
            similar_pairs = []
            if 'pre_nms_candidates' in detections:
                similar_pairs = self._find_similar_appearance_detections(detections['pre_nms_candidates'])
            
            # 轉換為 API 輸出格式
            results = []
            for i in range(len(detections['xyxy'])):
                x1, y1, x2, y2 = detections['xyxy'][i]
                class_id = int(detections['class_id'][i])
                confidence = float(detections['confidence'][i])
                
                # 獲取英文藥名
                english_name = self.class_names[class_id] if self.class_names and class_id < len(self.class_names) else f'Class_{class_id}'
                
                # 轉換為中文藥名
                chinese_name = CHINESE_DRUG_NAMES.get(english_name, english_name)
                
                results.append({
                    'class_id': class_id,
                    'class_name': chinese_name,
                    'class_name_en': english_name,
                    'class_name_zh': chinese_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
            
            # 返回結果時包含相似外觀檢測信息
            return {
                'detections': results,
                'similar_appearance_pairs': similar_pairs
            }
                
        except Exception as e:
            logger.error(f"❌ 後處理失敗: {e}")
            return []
            
    def _postprocess_detections(self, logits, boxes, threshold=0.7, top_k=30):
        """
        後處理核心實現 - 每位置最高分類 + NMS 策略
        
        流程: sigmoid激活 → 每位置最高分類 → 閾值過濾 → NMS → Top-K選擇 → 座標轉換
        
        Args:
            logits: (1, 300, num_classes) 類別預測
            boxes: (1, 300, 4) 框預測，cxcywh格式
            threshold: 信心度閾值
            top_k: 最大檢測數量
            
        Returns:
            Dict: 包含 'xyxy', 'confidence', 'class_id' 的檢測結果
        """
        # 移除批次維度
        logits = logits[0]  # (300, num_classes)
        boxes = boxes[0]    # (300, 4)
        
        # 計算信心度（使用 sigmoid）
        scores = 1 / (1 + np.exp(-logits))  # sigmoid 函數
        
        # 步驟1：每個位置取最高分類
        max_scores = np.max(scores, axis=1)  # (300,) 每個位置的最高分數
        max_classes = np.argmax(scores, axis=1)  # (300,) 每個位置的最佳類別
        
        # 步驟2：閾值過濾
        keep_mask = max_scores > threshold
        if not np.any(keep_mask):
            return {'xyxy': np.array([]).reshape(0, 4), 'confidence': np.array([]), 'class_id': np.array([])}
            
        filtered_scores = max_scores[keep_mask]
        filtered_classes = max_classes[keep_mask]
        filtered_boxes = boxes[keep_mask]
        
        # 步驟3：座標轉換 (在NMS之前進行，因為NMS需要xyxy格式)
        xyxy_boxes = CoordinateUtils.cxcywh_to_xyxy(filtered_boxes)
        target_size = INPUT_SIZE[0]  # 使用配置中的尺寸（width = height）
        scaled_boxes = xyxy_boxes * target_size
        
        # 步驟4：保存 NMS 前的候選檢測（用於相似外觀分析）
        pre_nms_candidates = {
            'boxes': scaled_boxes.copy(),
            'scores': filtered_scores.copy(), 
            'classes': filtered_classes.copy()
        }
        
        # 步驟5：NMS (非極大值抑制) - 移除重複檢測
        from .config import NMS_IOU_THRESHOLD
        nms_boxes, nms_scores, nms_classes = CoordinateUtils.non_max_suppression(
            scaled_boxes, filtered_scores, filtered_classes, 
            iou_threshold=NMS_IOU_THRESHOLD
        )
        
        # 步驟6：Top-K 選擇（在NMS後的結果中選擇）
        if len(nms_scores) > top_k:
            top_indices = np.argsort(nms_scores)[-top_k:]
            final_boxes = nms_boxes[top_indices]
            final_scores = nms_scores[top_indices]
            final_classes = nms_classes[top_indices]
        else:
            final_boxes = nms_boxes
            final_scores = nms_scores
            final_classes = nms_classes
        
        return {
            'xyxy': final_boxes,
            'confidence': final_scores,
            'class_id': final_classes,
            'pre_nms_candidates': pre_nms_candidates  # 新增：NMS 前的候選檢測
        }
    
    def _find_similar_appearance_detections(self, pre_nms_candidates: Dict) -> List[Tuple[int, int]]:
        """
        檢測相似外觀情況：同位置有多個不同類別且信心度相近的檢測
        
        Args:
            pre_nms_candidates: NMS 前的候選檢測，包含 'boxes', 'scores', 'classes'
            
        Returns:
            List[Tuple[int, int]]: 相似外觀檢測對的索引列表
        """
        from .config import SIMILAR_POSITION_IOU_THRESHOLD, SIMILAR_CONFIDENCE_THRESHOLD
        from .utils.coordinate_utils import CoordinateUtils
        
        boxes = pre_nms_candidates['boxes']
        scores = pre_nms_candidates['scores'] 
        classes = pre_nms_candidates['classes']
        
        similar_pairs = []
        
        # 逐對比較所有候選檢測
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                # 檢查是否為不同類別
                if classes[i] != classes[j]:
                    # 計算位置相似性 (IoU)
                    iou = CoordinateUtils.calculate_iou(boxes[i], boxes[j])
                    
                    if iou > SIMILAR_POSITION_IOU_THRESHOLD:
                        # 檢查信心度相似性
                        confidence_diff = abs(scores[i] - scores[j])
                        
                        if confidence_diff < SIMILAR_CONFIDENCE_THRESHOLD:
                            similar_pairs.append((i, j))
        
        return similar_pairs
        
    def get_classes(self) -> List[Dict[str, str]]:
        """
        取得所有支援的藥丸類別名稱（包含中英文）
        
        從COCO標註文件載入的類別清單，用於API端點 /classes
        
        Returns:
            List[Dict[str, str]]: 藥丸類別名稱列表，依照COCO類別ID排序
                                 每個項目包含英文名稱和中文名稱
                                 例如: [{'english': 'Amoxicillin', 'chinese': '安莫西林膠囊'}, ...]
        """
        if not self.class_names:
            return []
        
        result = []
        for class_name in self.class_names:
            # 跳過背景類別
            if class_name == "objects-P70T-danO":
                continue
            
            result.append({
                "english": class_name,
                "chinese": CHINESE_DRUG_NAMES.get(class_name, class_name)
            })
        
        return result
        
    def is_ready(self) -> bool:
        """
        檢查檢測器是否已完成初始化
        
        驗證ONNX模型和類別名稱是否都已成功載入，
        確保檢測器可以正常執行推理任務。
        
        Returns:
            bool: True表示檢測器已就緒，False表示仍在初始化中
        """
        return self.onnx_session is not None and self.class_names is not None
    
