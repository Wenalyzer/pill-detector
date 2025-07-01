"""
Docker 環境專用字體工具 - 簡化版
只處理 Linux 容器環境的字體載入
"""
import logging
from typing import Dict
from PIL import ImageFont

logger = logging.getLogger(__name__)

# Docker 容器中的單一字體路徑
DOCKER_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

class DockerFontManager:
    """Docker 環境專用字體管理器"""
    
    def __init__(self):
        self._font_cache: Dict[int, ImageFont.ImageFont] = {}
        self._font_available = self._check_font_availability()
    
    def _check_font_availability(self) -> bool:
        """檢查字體是否可用"""
        try:
            ImageFont.truetype(DOCKER_FONT_PATH, 12)
            logger.info(f"✅ Docker 字體可用: {DOCKER_FONT_PATH}")
            return True
        except Exception:
            logger.warning("⚠️ Docker 字體不可用，將使用預設字體")
            return False
    
    def get_font(self, font_size: int) -> ImageFont.ImageFont:
        """獲取字體"""
        if font_size in self._font_cache:
            return self._font_cache[font_size]
        
        # 嘗試載入指定字體
        if self._font_available:
            try:
                font = ImageFont.truetype(DOCKER_FONT_PATH, font_size)
                self._font_cache[font_size] = font
                return font
            except Exception as e:
                logger.warning(f"字體載入失敗: {e}")
        
        # 備援：預設字體
        font = ImageFont.load_default()
        self._font_cache[font_size] = font
        return font

# 全域實例
_font_manager = DockerFontManager()

def get_font(font_size: int) -> ImageFont.ImageFont:
    """獲取字體的便捷函數"""
    return _font_manager.get_font(font_size)