"""
檢測服務 - 整合 API 業務邏輯層
負責協調檢測器、標註器，處理不同輸入源，提供統一的檢測服務
"""
import logging
import base64
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple
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
            self._log_detection_result("URL", result['total_detections'], total_time, download_time=download_time)
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
            self._log_detection_result("檔案", result['total_detections'], total_time, load_time=load_time)
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
            download_start = time.perf_counter()
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            download_time = time.perf_counter() - download_start
            
            # 檢查內容類型
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"⚠️ 可疑的內容類型: {content_type}")
            
            # 載入圖像
            image = Image.open(BytesIO(response.content))
            
            # 統一處理圖像模式
            image = self._ensure_rgb_mode(image, f"📥 成功下載圖像 ({download_time*1000:.1f}ms)")
            return image
            
        except requests.RequestException as e:
            logger.error(f"圖像下載失敗: {str(e)}")
            raise Exception("圖像下載失敗")
        except (IOError, OSError) as e:
            logger.error(f"圖像載入失敗: {str(e)}")
            raise Exception("圖像格式不支援或已損壞")
        except Exception as e:
            logger.error(f"圖像處理未知錯誤: {str(e)}")
            raise Exception("圖像處理失敗")
    
    async def _load_image_from_bytes(self, file_content: bytes) -> Image.Image:
        """從字節內容載入圖像"""
        try:
            image = Image.open(BytesIO(file_content))
            
            # 統一處理圖像模式
            image = self._ensure_rgb_mode(image, "📁 成功載入圖像")
            return image
            
        except (IOError, OSError) as e:
            logger.error(f"檔案圖像載入失敗: {str(e)}")
            raise Exception("上傳的檔案格式不支援或已損壞")
        except Exception as e:
            logger.error(f"檔案處理未知錯誤: {str(e)}")
            raise Exception("檔案處理失敗")
    
    async def _detect_and_annotate(self, image: Image.Image) -> Dict:
        """執行檢測和標註的核心邏輯"""
        start_time = time.perf_counter()
        try:
            logger.debug(f"🖼️ 處理圖像尺寸: {image.size}")
            
            # 執行檢測（使用處理後的560x560圖片）
            detections, processed_image, inference_time, preprocess_time, postprocess_time, similar_pairs = await self._perform_detection(image)
            
            # 執行標註（座標和圖片完全匹配）
            annotation_start = time.perf_counter()
            annotated_image, label_areas = self.annotator.annotate_image(processed_image, detections)
            annotation_time = time.perf_counter() - annotation_start
            
            total_time = time.perf_counter() - start_time
            self._log_performance_breakdown(preprocess_time, inference_time, postprocess_time, annotation_time, total_time)
            
            # 分析檢測品質（包含相似外觀檢測）
            quality_analysis = self._analyze_detection_quality(detections, similar_pairs)
            
            return {
                'detections': detections,
                'annotated_image': f"data:image/{OUTPUT_IMAGE_FORMAT.lower()};base64,{self._image_to_base64(annotated_image)}",
                'total_detections': len(detections),
                'label_areas': label_areas,  # 開發用
                'image_info': {
                    'original_size': image.size,
                    'mode': image.mode
                },
                'quality_analysis': quality_analysis
            }
            
        except (IOError, OSError) as e:
            logger.error(f"❌ 圖像處理失敗: {e}", exc_info=True)
            raise Exception("圖像處理失敗")
        except RuntimeError as e:
            logger.error(f"❌ 模型推理失敗: {e}", exc_info=True)
            raise Exception("模型推理失敗")
        except Exception as e:
            logger.error(f"❌ 檢測和標註失敗: {e}", exc_info=True)
            raise Exception("檢測服務內部錯誤")
    
    async def _perform_detection(self, image: Image.Image) -> Tuple[list, Image.Image, float, float, float, list]:
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
        if (hasattr(self.detector, 'onnx_session') and 
            self.detector.onnx_session and 
            hasattr(self.detector.onnx_session, 'end_profiling')):
            try:
                # 只有在 profiling 真正啟用時才嘗試獲取報告
                prof_file = self.detector.onnx_session.end_profiling()
                if prof_file:
                    logger.debug(f"🔍 ONNX 性能分析報告: {prof_file}")
            except Exception as e:
                logger.debug(f"⚠️ 性能分析報告獲取失敗: {e}")  # 如果沒有啟用 profiling，忽略錯誤
        
        # 後處理（圖像已統一為INPUT_SIZE配置尺寸，無需傳遞尺寸）
        postprocess_start = time.perf_counter()
        detection_results = self.detector.postprocess_results(outputs)
        postprocess_time = time.perf_counter() - postprocess_start
        
        # 處理新的返回格式
        if isinstance(detection_results, dict):
            detections = detection_results['detections']
            similar_pairs = detection_results.get('similar_appearance_pairs', [])
        else:
            # 向後相容性
            detections = detection_results
            similar_pairs = []
        
        logger.debug(f"🎯 檢測到 {len(detections)} 個目標")
        return detections, processed_image, inference_time, preprocess_time, postprocess_time, similar_pairs
    
    def _image_to_base64(self, image: Image.Image, format: str = OUTPUT_IMAGE_FORMAT, quality: int = OUTPUT_IMAGE_QUALITY) -> str:
        """將圖像轉換為 base64 字符串"""
        buffer = BytesIO()
        image.save(buffer, format=format, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _ensure_rgb_mode(self, image: Image.Image, log_prefix: str) -> Image.Image:
        """統一處理圖像 RGB 轉換"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"{log_prefix}: {image.size}, 已轉換為 RGB")
        else:
            logger.info(f"{log_prefix}: {image.size}, {image.mode}")
        return image
    
    def _log_detection_result(self, detection_type: str, count: int, total_time: float, 
                            download_time: float = None, load_time: float = None):
        """統一日誌格式化檢測結果"""
        extra_info = ""
        if download_time is not None:
            extra_info = f" | 下載: {download_time:.2f}s"
        elif load_time is not None:
            extra_info = f" | 載入: {load_time:.2f}s"
        
        logger.info(f"✅ {detection_type}檢測完成: {count} 個藥丸{extra_info} | 總時間: {total_time:.2f}s")
    
    def _log_performance_breakdown(self, preprocess: float, inference: float, 
                                 postprocess: float, annotation: float, total: float):
        """統一日誌格式化性能分解"""
        logger.info(f"⚡ 預處理: {preprocess:.2f}s | 推理: {inference:.2f}s | "
                   f"後處理: {postprocess:.2f}s | 標註: {annotation:.2f}s | 總計: {total:.2f}s")
    
    def _analyze_detection_quality(self, detections: List[Dict], similar_pairs: List[Tuple[int, int]] = None) -> Dict:
        """
        分析檢測品質並提供重拍建議
        
        Args:
            detections: 檢測結果列表
            similar_pairs: 相似外觀檢測對列表
            
        Returns:
            品質分析結果，包含是否建議重拍和具體建議
        """
        if not detections:
            return {
                'should_retake': True,
                'reason': 'no_detection',
                'message': '未檢測到任何藥丸，建議重新拍攝',
                'suggestions': [
                    '確保藥丸清晰可見',
                    '改善光線條件',
                    '調整拍攝角度或距離'
                ]
            }
        
        # 優先檢查相似外觀情況
        if similar_pairs and len(similar_pairs) > 0:
            return {
                'should_retake': True,
                'reason': 'similar_appearance',
                'message': f'檢測到 {len(similar_pairs)} 組外觀相似的藥丸，建議重新拍攝以便更清晰區分',
                'suggestions': [
                    '調整拍攝距離，找到最清晰的對焦點',
                    '嘗試翻轉藥丸，拍攝有印記或文字的一面（如果有的話）',
                    '保持手機穩定，避免手震造成模糊',
                    '調整光線角度，讓藥丸表面更清晰可見'
                ],
                'similar_appearance_items': similar_pairs,
                'quality_score': max(0.3, 0.8 - len(similar_pairs) * 0.1)  # 根據相似外觀數量降低分數
            }
        
        # 檢查低信心度檢測
        low_confidence_count = 0
        uncertain_detections = []
        
        for i, detection in enumerate(detections):
            if detection['confidence'] < 0.7:  # 信心度 < 0.7 算不確定
                low_confidence_count += 1
                uncertain_detections.append(i + 1)
        
        # 如果超過一半檢測信心度不高，建議重拍
        total_detections = len(detections)
        low_confidence_ratio = low_confidence_count / total_detections
        
        if low_confidence_ratio > 0.5:
            return {
                'should_retake': True,
                'reason': 'low_confidence',
                'message': f'有 {low_confidence_count} 個藥丸的識別信心度較低，建議重新拍攝',
                'suggestions': [
                    '調整拍攝距離，找到最清晰的對焦點',
                    '嘗試翻轉藥丸，確認是否有更清晰的印記面',
                    '避免手震，保持拍攝穩定',
                    '確保光線充足，避免陰影遮擋'
                ],
                'uncertain_items': uncertain_detections,
                'quality_score': 1.0 - low_confidence_ratio
            }
        elif low_confidence_count > 0:
            # 有少量低信心度檢測，給予提醒但不強制重拍
            return {
                'should_retake': False,
                'reason': 'partial_uncertainty',
                'message': f'檢測品質良好，但有 {low_confidence_count} 個藥丸的信心度較低',
                'suggestions': [
                    '可考慮重新拍攝以提高識別準確度'
                ],
                'uncertain_items': uncertain_detections,
                'quality_score': 1.0 - low_confidence_ratio
            }
        
        return {
            'should_retake': False,
            'reason': 'good_quality',
            'message': '檢測品質良好，識別結果可信',
            'quality_score': 1.0 - low_confidence_ratio
        }
    
    def get_service_info(self) -> Dict:
        """獲取服務資訊"""
        return {
            'service_name': 'PillDetectionService',
            'version': API_VERSION,
            'status': 'ready' if self.is_ready() else 'initializing',
            'supported_classes': len(self.get_classes()),
            'components': {
                'detector': 'PillDetector',
                'annotator': 'ImageAnnotator',
                'model_loaded': self.detector.onnx_session is not None,
                'classes_loaded': self.detector.class_names is not None
            }
        }