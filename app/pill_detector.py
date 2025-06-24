"""
藥丸檢測核心模組：整合圖像預處理、模型推理與結果標註
"""
import os
import json
import logging
import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import requests
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from .config import *

logger = logging.getLogger(__name__)

class PillDetector:
    """藥丸檢測器主類"""
    
    def __init__(self):
        self.onnx_session = None
        self.class_names = None
        
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
            
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """圖像預處理：流程與 main_legacy.py 完全一致"""
        try:
            # 步驟1: 轉換為 CHW 並正規化到 [0,1]
            tensor_like = image_array.transpose((2, 0, 1)).astype(np.float32) / 255.0
            
            # 步驟2: ImageNet 正規化
            means = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(-1, 1, 1)
            stds = np.array(IMAGENET_STD, dtype=np.float32).reshape(-1, 1, 1)
            normalized = (tensor_like - means) / stds
            
            # 步驟3: 調整到模型輸入尺寸 (使用 OpenCV 以匹配原始流程)
            hwc_normalized = normalized.transpose((1, 2, 0))  # CHW -> HWC for OpenCV
            resized_hwc = cv2.resize(hwc_normalized, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            resized_chw = resized_hwc.transpose((2, 0, 1))  # HWC -> CHW
            
            # 步驟4: 添加 batch 維度
            batched = np.expand_dims(resized_chw, axis=0)
            
            return batched
            
        except Exception as e:
            logger.error(f"❌ 圖像預處理失敗: {e}")
            raise
        
    def postprocess_results(self, outputs, original_size: Tuple[int, int]) -> List[Dict]:
        """模型推理結果後處理，產生標準化檢測結果"""
        try:
            # 提取預測結果 - 按照 main_legacy.py 的方式
            pred_boxes, pred_logits = outputs[0][0], outputs[1][0]  # 移除 batch 維度
            
            # Sigmoid 激活
            prob = 1.0 / (1.0 + np.exp(-pred_logits))
            
            # Top-K 選擇
            prob_flat = prob.reshape(-1)
            topk_indices = np.argpartition(prob_flat, -TOP_K)[-TOP_K:]
            topk_indices = topk_indices[np.argsort(prob_flat[topk_indices])[::-1]]
            
            scores = prob_flat[topk_indices]
            topk_boxes = topk_indices // pred_logits.shape[1]
            labels = topk_indices % pred_logits.shape[1]
            
            # 邊界框格式轉換 cxcywh -> xyxy (向量化操作)
            def box_cxcywh_to_xyxy(boxes):
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1, y1 = cx - 0.5 * w, cy - 0.5 * h
                x2, y2 = cx + 0.5 * w, cy + 0.5 * h
                return np.stack([x1, y1, x2, y2], axis=1)
            
            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            selected_boxes = boxes_xyxy[topk_boxes]
            
            # 縮放到原始尺寸
            img_w, img_h = original_size
            scale_fct = np.array([img_w, img_h, img_w, img_h])
            final_boxes = selected_boxes * scale_fct
            
            # 應用閾值過濾
            valid_mask = scores >= CONFIDENCE_THRESHOLD
            if not np.any(valid_mask):
                logger.info("ℹ️ 沒有檢測結果超過閾值")
                return []
            
            valid_boxes = final_boxes[valid_mask]
            valid_confidences = scores[valid_mask]
            valid_class_ids = labels[valid_mask]
            
            # 座標範圍限制
            valid_boxes[:, [0, 2]] = np.clip(valid_boxes[:, [0, 2]], 0, img_w)
            valid_boxes[:, [1, 3]] = np.clip(valid_boxes[:, [1, 3]], 0, img_h)
            
            # 轉換為 API 輸出格式
            results = []
            for i in range(len(valid_boxes)):
                x1, y1, x2, y2 = valid_boxes[i]
                results.append({
                    'class_id': int(valid_class_ids[i]),
                    'class_name': self.class_names[valid_class_ids[i]] if self.class_names else f'Class_{valid_class_ids[i]}',
                    'confidence': float(valid_confidences[i]),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                    
            return results
                
        except Exception as e:
            logger.error(f"❌ 後處理失敗: {e}")
            return []
            
    def get_optimal_font(self, font_size: int):
        """取得最佳可用字體（與 main_legacy.py 相同邏輯）"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                continue
        
        return ImageFont.load_default()

    def calculate_smart_label_positions(self, detections: List[Dict], image_size: Tuple[int, int], font_size: int) -> List[Optional[Tuple[int, int]]]:
        """智能計算標籤位置，避免重疊與遮擋（與 main_legacy.py 一致）"""
        w, h = image_size
        
        if len(detections) == 0:
            return []
        
        label_height = font_size + 8
        max_label_width = font_size * 12
        
        label_positions = []
        occupied_regions = []
        
        # 按信心度排序
        sorted_detections = sorted(enumerate(detections), key=lambda x: x[1]['confidence'], reverse=True)
        
        for original_idx, det in sorted_detections:
            bbox = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            
            label_text = f"{class_name} {conf:.2f}"
            label_width = min(len(label_text) * font_size * 0.6, max_label_width)
            
            x1, y1, x2, y2 = bbox
            
            # 候選位置 - 增加更多選項
            candidates = [
                # 上方 - 多個位置
                (x1, y1 - label_height - 5, x1 + label_width, y1 - 5),
                (x1 + (x2-x1)//4, y1 - label_height - 5, x1 + (x2-x1)//4 + label_width, y1 - 5),
                (max(0, x2 - label_width), y1 - label_height - 5, x2, y1 - 5),
                
                # 下方 - 多個位置  
                (x1, y2 + 5, x1 + label_width, y2 + 5 + label_height),
                (x1 + (x2-x1)//4, y2 + 5, x1 + (x2-x1)//4 + label_width, y2 + 5 + label_height),
                (max(0, x2 - label_width), y2 + 5, x2, y2 + 5 + label_height),
                
                # 左側
                (x1 - label_width - 5, y1, x1 - 5, y1 + label_height),
                (x1 - label_width - 5, y1 + (y2-y1)//4, x1 - 5, y1 + (y2-y1)//4 + label_height),
                
                # 右側
                (x2 + 5, y1, x2 + 5 + label_width, y1 + label_height),
                (x2 + 5, y1 + (y2-y1)//4, x2 + 5 + label_width, y1 + (y2-y1)//4 + label_height),
                
                # 框內上方
                (x1 + 5, y1 + 5, x1 + 5 + label_width, y1 + 5 + label_height),
                # 框內下方
                (x1 + 5, y2 - label_height - 5, x1 + 5 + label_width, y2 - 5),
            ]
            
            # 找最佳位置
            best_position = None
            for pos_x1, pos_y1, pos_x2, pos_y2 in candidates:
                # 1. 邊界檢查
                if pos_x1 < 0 or pos_y1 < 0 or pos_x2 > w or pos_y2 > h:
                    continue
                
                # 2. 與其他標籤重疊檢查
                overlaps_label = False
                for occupied in occupied_regions:
                    if not (pos_x2 < occupied[0] or pos_x1 > occupied[2] or 
                           pos_y2 < occupied[1] or pos_y1 > occupied[3]):
                        overlaps_label = True
                        break
                
                if overlaps_label:
                    continue
                
                # 3. 檢查是否遮擋其他檢測框
                blocks_other_boxes = False
                for other_det in detections:
                    if other_det == det:  # 跳過自己
                        continue
                        
                    ox1, oy1, ox2, oy2 = other_det['bbox']
                    
                    # 計算重疊面積
                    overlap_x1 = max(pos_x1, ox1)
                    overlap_y1 = max(pos_y1, oy1)
                    overlap_x2 = min(pos_x2, ox2)
                    overlap_y2 = min(pos_y2, oy2)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        box_area = (ox2 - ox1) * (oy2 - oy1)
                        
                        # 如果標籤遮擋其他框超過 20%，則不合適
                        if overlap_area > box_area * 0.2:
                            blocks_other_boxes = True
                            break
                
                if not blocks_other_boxes:
                    best_position = (pos_x1, pos_y1, pos_x2, pos_y2)
                    break
            
            # 備援位置 - 如果都不行，尋找圖像邊緣空白區域
            if best_position is None:
                # 嘗試圖像四個角落
                corner_candidates = [
                    # 左上角
                    (5, 5, 5 + label_width, 5 + label_height),
                    # 右上角  
                    (w - label_width - 5, 5, w - 5, 5 + label_height),
                    # 左下角
                    (5, h - label_height - 5, 5 + label_width, h - 5),
                    # 右下角
                    (w - label_width - 5, h - label_height - 5, w - 5, h - 5),
                ]
                
                for pos_x1, pos_y1, pos_x2, pos_y2 in corner_candidates:
                    # 檢查角落位置是否與已有標籤重疊
                    corner_overlap = False
                    for occupied in occupied_regions:
                        if not (pos_x2 < occupied[0] or pos_x1 > occupied[2] or 
                               pos_y2 < occupied[1] or pos_y1 > occupied[3]):
                            corner_overlap = True
                            break
                    
                    if not corner_overlap:
                        best_position = (pos_x1, pos_y1, pos_x2, pos_y2)
                        break
                
                # 最終備援：強制放在框上方（即使可能重疊）
                if best_position is None:
                    pos_x1 = max(5, min(x1, w - label_width - 5))
                    pos_y1 = max(label_height + 5, y1 - 10)
                    best_position = (pos_x1, pos_y1, pos_x1 + label_width, pos_y1 + label_height)
            
            # 記錄位置
            label_positions.append((best_position[0], best_position[1] + label_height))
            occupied_regions.append(best_position)
        
        # 重新排序到原始順序
        final_positions: List[Optional[Tuple[int, int]]] = [None] * len(detections)
        for i, (original_idx, _) in enumerate(sorted_detections):
            final_positions[original_idx] = label_positions[i]
        
        return final_positions

    def annotate_image(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """在圖像上標註檢測結果（採用智能標籤定位）"""
        if not detections:
            return image
            
        # 創建副本以避免修改原圖
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # 獲取最佳字體
        font = self.get_optimal_font(FONT_SIZE)
        
        # 計算智能標籤位置
        label_positions = self.calculate_smart_label_positions(detections, image.size, FONT_SIZE)
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 選擇顏色
            color = COLORS[i % len(COLORS)]
            
            # 繪製邊界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_THICKNESS)
            
            # 準備標籤文字
            label = f"{class_name} {confidence:.2f}"
            
            # 使用智能計算的位置
            if i < len(label_positions) and label_positions[i] is not None:
                position = label_positions[i]
                if position is not None:
                    text_x, text_y = position
                    text_y -= FONT_SIZE  # 調整為文字頂部位置
                else:
                    # 備援位置
                    text_x = x1
                    text_y = y1 - FONT_SIZE - 5
                    if text_y < 0:
                        text_y = y1 + 5
            else:
                # 備援位置
                text_x = x1
                text_y = y1 - FONT_SIZE - 5
                if text_y < 0:
                    text_y = y1 + 5
            
            # 計算文字尺寸
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            padding = 4
            
            # 繪製標籤背景
            draw.rectangle(
                [text_x - padding, text_y - padding, 
                 text_x + text_width + padding, text_y + text_height + padding],
                fill=color, outline=color
            )
            
            # 繪製標籤文字
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
            
        return annotated
        
    async def detect_from_url(self, url: str) -> Dict:
        """從圖片 URL 進行藥丸檢測"""
        try:
            # 下載圖像
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 載入並處理圖像
            image = Image.open(BytesIO(response.content))
            return await self._detect_from_image(image)
            
        except Exception as e:
            logger.error(f"❌ URL 檢測失敗: {e}")
            raise
            
    async def detect_from_file(self, file_content: bytes) -> Dict:
        """從上傳檔案內容進行藥丸檢測"""
        try:
            image = Image.open(BytesIO(file_content))
            return await self._detect_from_image(image)
            
        except Exception as e:
            logger.error(f"❌ 檔案檢測失敗: {e}")
            raise
            
    async def _detect_from_image(self, image: Image.Image) -> Dict:
        """執行單張圖片的完整檢測流程"""
        # 轉換為 RGB 模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        original_size = image.size
        
        # 轉換為 numpy 數組
        image_array = np.array(image)
        
        # 預處理 (包含 resize)
        input_tensor = self.preprocess_image(image_array)
        
        # 模型推理
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_tensor})
        
        # 後處理
        detections = self.postprocess_results(outputs, original_size)
        
        # 標註圖像
        annotated_image = self.annotate_image(image, detections)
        
        # 轉換為 base64
        buffer = BytesIO()
        annotated_image.save(buffer, format='JPEG', quality=95)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'detections': detections,
            'annotated_image': f"data:image/jpeg;base64,{image_base64}",
            'total_detections': len(detections)
        }
        
    def get_classes(self) -> List[str]:
        """取得所有支援的藥丸類別名稱"""
        return self.class_names or []
        
    def is_ready(self) -> bool:
        """檢查模型與類別名稱是否載入完成"""
        return self.onnx_session is not None and self.class_names is not None