"""
座標轉換工具 - 處理檢測框座標格式轉換
純函數實現，無狀態，易於測試和重用
"""
import numpy as np

class CoordinateUtils:
    """座標轉換工具類 - 所有方法都是靜態方法"""
    
    @staticmethod
    def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """
        中心點格式轉角點格式
        
        Args:
            boxes: (N, 4) [cx, cy, w, h] 歸一化座標 [0, 1]
            
        Returns:
            (N, 4) [x1, y1, x2, y2] 歸一化座標 [0, 1]
        """
        if boxes.size == 0:
            return boxes
            
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return np.stack([x1, y1, x2, y2], axis=1)
    
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        計算兩個檢測框的IoU (Intersection over Union)
        
        Args:
            box1: [x1, y1, x2, y2] 格式的檢測框
            box2: [x1, y1, x2, y2] 格式的檢測框
            
        Returns:
            IoU 值 (0-1)
        """
        # 計算交集區域
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # 檢查是否有交集
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        # 計算交集面積
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 計算兩個框的面積
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 計算聯合面積
        union_area = area1 + area2 - inter_area
        
        # 避免除零
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                          classes: np.ndarray, iou_threshold: float = 0.5) -> tuple:
        """
        非極大值抑制 (NMS) - 移除重複的檢測框
        
        Args:
            boxes: (N, 4) [x1, y1, x2, y2] 檢測框座標
            scores: (N,) 信心度分數
            classes: (N,) 類別ID
            iou_threshold: IoU閾值，超過此值的重複框會被移除
            
        Returns:
            (filtered_boxes, filtered_scores, filtered_classes): 過濾後的結果
        """
        if len(boxes) == 0:
            return boxes, scores, classes
        
        # 按信心度降序排序
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # 取信心度最高的框
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # 計算當前框與其他框的IoU
            current_box = boxes[current_idx]
            remaining_indices = sorted_indices[1:]
            
            # 找出IoU小於閾值的框（保留）
            keep_mask = []
            for idx in remaining_indices:
                other_box = boxes[idx]
                iou = CoordinateUtils.calculate_iou(current_box, other_box)
                if iou < iou_threshold:
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            
            # 更新剩餘的索引
            sorted_indices = remaining_indices[np.array(keep_mask)]
        
        # 返回保留的檢測結果
        keep_indices = np.array(keep_indices)
        return boxes[keep_indices], scores[keep_indices], classes[keep_indices]
