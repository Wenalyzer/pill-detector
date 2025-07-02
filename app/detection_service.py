"""
檢測服務 - 整合 API 業務邏輯層
負責協調檢測器、標註器，處理不同輸入源，提供統一的檢測服務
"""
import logging
import base64
import time
from io import BytesIO
from typing import Dict, Optional, Tuple
import requests
import numpy as np
from PIL import Image

from .pill_detector import PillDetector
from .image_annotator import ImageAnnotator
from .config import *

logger = logging.getLogger(__name__)


class DetectionService:
    """
    檢測服務 - API 業務邏輯整合層
    
    職責：
    - 協調檢測器和標註器
    - 處理不同輸入源 (URL, 檔案)
    - 統一錯誤處理和日誌
    - 提供高階 API 接口
    """
    
    def __init__(self):
        """初始化檢測服務"""
        self.detector = PillDetector()
        self.annotator = ImageAnnotator()
        self._onnx_input_name = None  # 快取 ONNX 輸入名稱
        
    async def initialize(self):
        """初始化檢測服務（載入模型等）"""
        await self.detector.initialize()
        # 快取 ONNX 輸入名稱
        if self.detector.onnx_session:
            self._onnx_input_name = self.detector.onnx_session.get_inputs()[0].name
        logger.info("✅ 檢測服務初始化完成")
    
    def is_ready(self) -> bool:
        """檢查服務是否就緒"""
        return self.detector.is_ready()
    
    def get_classes(self) -> list:
        """獲取支援的藥物類別"""
        return self.detector.get_classes()
    
    async def detect_from_url(self, url: str) -> Dict:
        """
        從圖片 URL 進行檢測
        
        Args:
            url: 圖片 URL
            
        Returns:
            檢測結果字典
            
        Raises:
            HTTPException: 網路錯誤或檢測失敗
        """
        start_time = time.perf_counter()
        try:
            logger.info(f"🔍 開始 URL 檢測: {url}")
            
            # 下載圖像
            download_start = time.perf_counter()
            image = await self._download_image_from_url(url)
            download_time = time.perf_counter() - download_start
            
            # 執行檢測
            result = await self._detect_and_annotate(image)
            
            total_time = time.perf_counter() - start_time
            logger.info(f"✅ URL 檢測完成: {result['total_detections']} 個藥丸 | "
                       f"下載: {download_time:.2f}s | 總時間: {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ URL 檢測失敗: {e}")
            raise
    
    async def detect_from_file(self, file_content: bytes, filename: str = "uploaded_image") -> Dict:
        """
        從上傳檔案進行檢測
        
        Args:
            file_content: 檔案內容
            filename: 檔案名稱（用於日誌）
            
        Returns:
            檢測結果字典
            
        Raises:
            HTTPException: 檔案處理錯誤或檢測失敗
        """
        start_time = time.perf_counter()
        try:
            file_size_mb = len(file_content) / (1024 * 1024)
            logger.info(f"🔍 開始檔案檢測: {filename} ({file_size_mb:.1f}MB)")
            
            # 載入圖像
            load_start = time.perf_counter()
            image = await self._load_image_from_bytes(file_content)
            load_time = time.perf_counter() - load_start
            
            # 執行檢測
            result = await self._detect_and_annotate(image)
            
            total_time = time.perf_counter() - start_time
            logger.info(f"✅ 檔案檢測完成: {result['total_detections']} 個藥丸 | "
                       f"載入: {load_time:.2f}s | 總時間: {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ 檔案檢測失敗: {e}")
            raise
    
    async def detect_unified(self, image_url: Optional[str] = None, 
                           file_content: Optional[bytes] = None,
                           filename: str = "image") -> Dict:
        """
        統一檢測接口 - 支援 URL 和檔案
        
        Args:
            image_url: 圖片 URL (可選)
            file_content: 檔案內容 (可選)
            filename: 檔案名稱
            
        Returns:
            檢測結果字典
        """
        if image_url:
            return await self.detect_from_url(image_url)
        elif file_content:
            return await self.detect_from_file(file_content, filename)
        else:
            raise ValueError("必須提供 image_url 或 file_content")
    
    async def _download_image_from_url(self, url: str) -> Image.Image:
        """從 URL 下載圖像"""
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # 檢查內容類型
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"⚠️ 可疑的內容類型: {content_type}")
            
            # 載入圖像
            image = Image.open(BytesIO(response.content))
            
            # 統一轉換為 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"📥 成功下載圖像: {image.size}, 已轉換為 RGB")
            else:
                logger.info(f"📥 成功下載圖像: {image.size}, {image.mode}")
            
            return image
            
        except requests.RequestException as e:
            raise Exception(f"圖像下載失敗: {e}")
        except (IOError, OSError) as e:
            raise Exception(f"圖像載入失敗: {e}")
        except Exception as e:
            raise Exception(f"圖像處理未知錯誤: {e}")
    
    async def _load_image_from_bytes(self, file_content: bytes) -> Image.Image:
        """從字節內容載入圖像"""
        try:
            image = Image.open(BytesIO(file_content))
            
            # 統一轉換為 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"📁 成功載入圖像: {image.size}, 已轉換為 RGB")
            else:
                logger.info(f"📁 成功載入圖像: {image.size}, {image.mode}")
            
            return image
            
        except (IOError, OSError) as e:
            raise Exception(f"檔案圖像載入失敗: {e}")
        except Exception as e:
            raise Exception(f"檔案處理未知錯誤: {e}")
    
    async def _detect_and_annotate(self, image: Image.Image) -> Dict:
        """執行檢測和標註的核心邏輯"""
        start_time = time.perf_counter()
        try:
            logger.debug(f"🖼️ 處理圖像尺寸: {image.size}")
            
            # 執行檢測（使用處理後的560x560圖片）
            detections, processed_image, inference_time, preprocess_time, postprocess_time = await self._perform_detection(image)
            
            # 執行標註（座標和圖片完全匹配）
            annotation_start = time.perf_counter()
            annotated_image, label_areas = self.annotator.annotate_image(processed_image, detections)
            annotation_time = time.perf_counter() - annotation_start
            
            total_time = time.perf_counter() - start_time
            logger.info(f"⚡ 預處理: {preprocess_time:.2f}s | 推理: {inference_time:.2f}s | "
                       f"後處理: {postprocess_time:.2f}s | 標註: {annotation_time:.2f}s | 總計: {total_time:.2f}s")
            
            return {
                'detections': detections,
                'annotated_image': f"data:image/{OUTPUT_IMAGE_FORMAT.lower()};base64,{self._image_to_base64(annotated_image)}",
                'total_detections': len(detections),
                'label_areas': label_areas,  # 開發用
                'image_info': {
                    'original_size': image.size,
                    'mode': image.mode
                }
            }
            
        except (IOError, OSError) as e:
            logger.error(f"❌ 圖像處理失敗: {e}")
            raise Exception(f"圖像處理失敗: {e}")
        except RuntimeError as e:
            logger.error(f"❌ 模型推理失敗: {e}")
            raise Exception(f"模型推理失敗: {e}")
        except Exception as e:
            logger.error(f"❌ 檢測和標註失敗: {e}")
            raise
    
    async def _perform_detection(self, image: Image.Image) -> Tuple[list, Image.Image, float, float, float]:
        """執行檢測，返回檢測結果、處理後的圖片和時間統計"""
        # 轉換為 numpy 數組
        image_array = np.array(image)
        
        # 預處理（返回tensor和處理後的圖片）
        preprocess_start = time.perf_counter()
        input_tensor, processed_image = self.detector.preprocess_image(image_array)
        preprocess_time = time.perf_counter() - preprocess_start
        
        # 模型推理
        inference_start = time.perf_counter()
        outputs = self.detector.onnx_session.run(None, {self._onnx_input_name: input_tensor})
        inference_time = time.perf_counter() - inference_start
        
        # 性能分析報告（如果啟用）
        if hasattr(self.detector, 'onnx_session') and self.detector.onnx_session:
            try:
                prof_file = self.detector.onnx_session.end_profiling()
                if prof_file:
                    logger.debug(f"🔍 ONNX 性能分析報告: {prof_file}")
            except:
                pass  # 如果沒有啟用 profiling，忽略錯誤
        
        # 後處理（圖像已統一為INPUT_SIZE配置尺寸，無需傳遞尺寸）
        postprocess_start = time.perf_counter()
        detections = self.detector.postprocess_results(outputs)
        postprocess_time = time.perf_counter() - postprocess_start
        
        logger.debug(f"🎯 檢測到 {len(detections)} 個目標")
        return detections, processed_image, inference_time, preprocess_time, postprocess_time
    
    def _image_to_base64(self, image: Image.Image, format: str = OUTPUT_IMAGE_FORMAT, quality: int = OUTPUT_IMAGE_QUALITY) -> str:
        """將圖像轉換為 base64 字符串"""
        buffer = BytesIO()
        image.save(buffer, format=format, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def get_service_info(self) -> Dict:
        """獲取服務資訊"""
        return {
            'service_name': 'PillDetectionService',
            'version': '2.0.0',
            'status': 'ready' if self.is_ready() else 'initializing',
            'supported_classes': len(self.get_classes()),
            'components': {
                'detector': 'PillDetector',
                'annotator': 'ImageAnnotator',
                'model_loaded': self.detector.onnx_session is not None,
                'classes_loaded': self.detector.class_names is not None
            }
        }