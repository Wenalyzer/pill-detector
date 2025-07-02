"""
藥丸檢測應用模組
Pillow 預處理實現，基於 RF-DETR ONNX 模型
"""

from .pill_detector import PillDetector
from .detection_service import DetectionService  
from .image_annotator import ImageAnnotator
from .config import *

__version__ = "2.0.0"  # 與 config.py 保持一致
__all__ = ["PillDetector", "DetectionService", "ImageAnnotator"]