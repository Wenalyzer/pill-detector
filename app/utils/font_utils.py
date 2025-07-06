"""
Docker 環境專用字體工具 - 簡化版
只處理 Linux 容器環境的字體載入
"""
import logging
from typing import Dict
from PIL import ImageFont

logger = logging.getLogger(__name__)

# Docker 容器中的中文字體路徑 (支援繁體中文)
DOCKER_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FALLBACK_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

class DockerFontManager:
    """Docker 環境專用字體管理器"""
    
    def __init__(self):
        self._font_cache: Dict[int, ImageFont.ImageFont] = {}
        self._chinese_font_available = self._check_font_availability()
    
    @property
    def supports_chinese(self) -> bool:
        """檢查是否支援中文"""
        return self._chinese_font_available
    
    def _check_font_availability(self) -> bool:
        """檢查中文字體是否可用"""
        try:
            ImageFont.truetype(DOCKER_FONT_PATH, 12)
            logger.info(f"✅ 中文字體可用: {DOCKER_FONT_PATH}")
            return True
        except Exception:
            try:
                ImageFont.truetype(FALLBACK_FONT_PATH, 12)
                logger.warning(f"⚠️ 中文字體不可用，使用備援字體: {FALLBACK_FONT_PATH}")
                return False
            except Exception:
                logger.warning("⚠️ 所有字體都不可用，將使用預設字體")
                return False
    
    def get_font(self, font_size: int) -> ImageFont.ImageFont:
        """獲取字體（優先中文字體）"""
        if font_size in self._font_cache:
            return self._font_cache[font_size]
        
        # 嘗試載入中文字體
        if self._chinese_font_available:
            try:
                font = ImageFont.truetype(DOCKER_FONT_PATH, font_size)
                self._font_cache[font_size] = font
                return font
            except Exception as e:
                logger.warning(f"中文字體載入失敗: {e}")
        
        # 備援：英文字體
        try:
            font = ImageFont.truetype(FALLBACK_FONT_PATH, font_size)
            self._font_cache[font_size] = font
            return font
        except Exception as e:
            logger.warning(f"備援字體載入失敗: {e}")
        
        # 最終備援：預設字體
        font = ImageFont.load_default()
        self._font_cache[font_size] = font
        return font

# 全域實例
_font_manager = DockerFontManager()

def get_font(font_size: int) -> ImageFont.ImageFont:
    """獲取字體的便捷函數"""
    return _font_manager.get_font(font_size)

def supports_chinese() -> bool:
    """檢查當前字體是否支援中文"""
    return _font_manager.supports_chinese