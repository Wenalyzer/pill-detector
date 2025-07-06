"""
圖像標註器 - 專注於檢測結果的可視化標註
負責標籤位置計算、圖像繪製等可視化功能
"""
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw

from .config import *
from .utils.font_utils import get_font, supports_chinese

logger = logging.getLogger(__name__)


class ImageAnnotator:
    """
    圖像標註器 - 專注於檢測結果可視化
    
    主要功能：
    - 標籤位置計算
    - 圖像標註繪製
    - 避免標籤重疊和遮擋
    """
    
    def __init__(self):
        """初始化標註器"""
        pass
    
    def annotate_image(self, image: Image.Image, detections: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """
        在圖像上標註檢測結果
        
        Args:
            image: 原始圖像
            detections: 檢測結果列表
            
        Returns:
            (標註後的圖像, 標籤位置列表)
        """
        if not detections:
            return image, []
            
        # 創建副本以避免修改原圖
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # 獲取字體
        font = get_font(FONT_SIZE)
        
        # 創建復用的文字尺寸計算器
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        
        # 第一步：繪製所有邊界框
        for i, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            color = COLORS[i % len(COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_THICKNESS)
        
        # 第二步：計算標籤位置（避免重疊）
        label_positions = self._calculate_non_overlapping_positions(detections, image.size, font, temp_draw)
        
        # 第三步：繪製所有標籤並記錄位置
        label_areas = []
        chinese_supported = supports_chinese()
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            confidence = det['confidence']
            color = COLORS[i % len(COLORS)]
            
            # 根據字體支援情況選擇顯示語言
            if chinese_supported and 'class_name_zh' in det:
                display_name = det['class_name_zh']
            elif 'class_name_en' in det:
                display_name = det['class_name_en']
            else:
                display_name = det.get('class_name', 'Unknown')
            
            # 準備標籤文字 (雙行格式)
            label = f"{display_name}\n{confidence:.2f}"
            
            # 使用計算好的位置
            text_x, text_y = label_positions[i]
            
            # 計算文字尺寸
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            padding = 6
            
            # 計算標籤區域
            bg_x1 = text_x - padding
            bg_y1 = text_y - padding
            bg_x2 = text_x + text_width + padding
            bg_y2 = text_y + text_height + padding
            
            # 記錄標籤位置（開發用）
            label_areas.append({
                'label_bbox': [bg_x1, bg_y1, bg_x2, bg_y2]
            })
            
            # 繪製標籤背景
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color, outline=color)
            
            # 繪製標籤文字
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
            
        return annotated, label_areas
    
    def _calculate_non_overlapping_positions(self, detections: List[Dict], image_size: Tuple[int, int], font, temp_draw) -> List[Tuple[int, int]]:
        """
        漸進式標籤位置計算 - 避免與檢測框和已放置標籤重疊
        """
        if not detections:
            return []
        
        w, h = image_size
        padding = 6
        chinese_supported = supports_chinese()
        
        # 提取所有檢測框區域
        detection_boxes = [det['bbox'] for det in detections]
        
        # 預計算圖像對角線長度（用於距離分數）
        max_distance = (w * w + h * h) ** 0.5
        
        # 儲存已放置的標籤區域
        placed_labels = []
        final_positions = []
        
        # 逐個處理每個標籤
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # 根據字體支援情況選擇顯示語言（與繪製邏輯保持一致）
            if chinese_supported and 'class_name_zh' in det:
                display_name = det['class_name_zh']
            elif 'class_name_en' in det:
                display_name = det['class_name_en']
            else:
                display_name = det.get('class_name', 'Unknown')
            
            # 計算標籤尺寸（復用temp_draw）
            label_text = f"{display_name}\n{confidence:.2f}"
            text_bbox = temp_draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 生成候選位置（基本四方向+角落位置）
            candidates = [
                # 基本四方向
                ("上方", x1, y1 - text_height - padding),
                ("下方", x1, y2 + padding),
                ("左側", x1 - text_width - padding, y1),
                ("右側", x2 + padding, y1),
                
                # 角落位置
                ("左上角", x1, y1 - text_height - padding),
                ("右上角", x2 - text_width, y1 - text_height - padding),
                ("左下角", x1, y2 + padding),
                ("右下角", x2 - text_width, y2 + padding)
            ]
            
            # 快速優先策略：先找完全無重疊的位置，再找最佳的
            perfect_positions = []  # 完全無重疊的位置
            acceptable_positions = []  # 可接受重疊的位置
            
            for _, text_x, text_y in candidates:
                # 計算標籤區域
                bg_x1 = text_x - padding
                bg_y1 = text_y - padding
                bg_x2 = text_x + text_width + padding
                bg_y2 = text_y + text_height + padding
                label_area = (bg_x1, bg_y1, bg_x2, bg_y2)
                
                # 檢查邊界
                if not (bg_x1 >= 0 and bg_y1 >= 0 and bg_x2 <= w and bg_y2 <= h):
                    continue
                
                # 快速檢查：計算最大單一重疊
                max_overlap = 0.0
                total_overlap = 0.0
                
                # 與其他檢測框的重疊（排除自己的檢測框）
                for j in range(len(detection_boxes)):
                    if j != i:
                        overlap = self._calculate_overlap_ratio(label_area, detection_boxes[j])
                        max_overlap = max(max_overlap, overlap)
                        total_overlap += overlap
                        
                        # 早期退出：如果重疊太嚴重直接跳過
                        if max_overlap > 0.1:
                            break
                
                if max_overlap > 0.1:
                    continue
                
                # 與已放置標籤的重疊
                for placed_label in placed_labels:
                    overlap = self._calculate_overlap_ratio(label_area, placed_label)
                    max_overlap = max(max_overlap, overlap)
                    total_overlap += overlap
                    
                    if max_overlap > 0.1:
                        break
                
                if max_overlap > 0.1:
                    continue
                
                # 計算距離分數（距離其他檢測框越遠越好）
                distance_score = self._calculate_distance_score(label_area, detection_boxes, i, max_distance)
                
                # 綜合分數：重疊懲罰 + 距離獎勵
                total_score = total_overlap - distance_score * 0.1  # 距離權重0.1
                
                # 分類位置
                position_data = (text_x, text_y, label_area, total_score)
                if total_overlap == 0.0:
                    perfect_positions.append(position_data)
                else:
                    acceptable_positions.append(position_data)
            
            # 選擇位置：優先完全無重疊，其次綜合分數最佳
            chosen_position = None
            if perfect_positions:
                # 有完全無重疊的位置，選擇距離最遠的
                perfect_positions.sort(key=lambda x: x[3])  # 按total_score排序（越小越好）
                text_x, text_y, label_area, _ = perfect_positions[0]
                chosen_position = (text_x, text_y)
                placed_labels.append(label_area)
            elif acceptable_positions:
                # 選擇綜合分數最佳的
                acceptable_positions.sort(key=lambda x: x[3])  # 按total_score排序
                text_x, text_y, label_area, _ = acceptable_positions[0]
                chosen_position = (text_x, text_y)
                placed_labels.append(label_area)
            
            # 如果找不到有效位置，使用強制位置
            if chosen_position is None:
                text_x = max(padding, min(x1, w - text_width - padding))
                text_y = max(padding, y1 - text_height - padding)
                bg_x1 = text_x - padding
                bg_y1 = text_y - padding
                bg_x2 = text_x + text_width + padding
                bg_y2 = text_y + text_height + padding
                
                chosen_position = (text_x, text_y)
                placed_labels.append((bg_x1, bg_y1, bg_x2, bg_y2))
            
            final_positions.append(chosen_position)
        
        return final_positions
    
    
    def _calculate_overlap_ratio(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        計算兩個矩形的重疊比例
        
        Returns:
            重疊比例（相對於較小矩形的面積）
        """
        x1, y1, x2, y2 = box1
        ox1, oy1, ox2, oy2 = box2
        
        # 如果完全不重疊
        if x2 <= ox1 or x1 >= ox2 or y2 <= oy1 or y1 >= oy2:
            return 0.0
        
        # 計算重疊區域
        overlap_x1 = max(x1, ox1)
        overlap_y1 = max(y1, oy1)
        overlap_x2 = min(x2, ox2)
        overlap_y2 = min(y2, oy2)
        
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        
        # 計算兩個矩形的面積
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (ox2 - ox1) * (oy2 - oy1)
        
        # 計算重疊比例（相對於較小的矩形）
        smaller_area = min(area1, area2)
        return overlap_area / smaller_area if smaller_area > 0 else 0.0
    
    def _calculate_distance_score(self, label_area: Tuple[int, int, int, int], detection_boxes: List[List[int]], current_index: int, max_distance: float) -> float:
        """
        計算標籤到其他檢測框的距離分數（距離越遠分數越高）
        
        Args:
            label_area: 標籤區域 (x1, y1, x2, y2)
            detection_boxes: 所有檢測框列表
            current_index: 當前檢測框的索引（排除自己）
            
        Returns:
            距離分數（0-1之間，越大表示距離越遠）
        """
        lx1, ly1, lx2, ly2 = label_area
        label_center_x = (lx1 + lx2) / 2
        label_center_y = (ly1 + ly2) / 2
        
        min_distance = float('inf')
        
        # 計算到其他檢測框的最小距離
        for j, det_box in enumerate(detection_boxes):
            if j == current_index:  # 跳過自己的檢測框
                continue
                
            dx1, dy1, dx2, dy2 = det_box
            det_center_x = (dx1 + dx2) / 2
            det_center_y = (dy1 + dy2) / 2
            
            # 計算中心點距離
            distance = ((label_center_x - det_center_x) ** 2 + (label_center_y - det_center_y) ** 2) ** 0.5
            min_distance = min(min_distance, distance)
        
        # 將距離轉換為0-1分數（距離越遠分數越高）
        distance_score = min(min_distance / max_distance, 1.0)
        
        return distance_score
