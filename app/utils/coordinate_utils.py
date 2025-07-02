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
    
    
