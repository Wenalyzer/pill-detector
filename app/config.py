"""
藥丸檢測 API 配置設定
"""

# API 基本設定
API_TITLE = "藥丸檢測 API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "使用 RF-DETR 模型進行藥丸識別的 API 服務"

# 檔案設定
MAX_FILE_SIZE = 25 * 1024 * 1024  
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# 模型設定
MODEL_PATH = "models/inference_model.onnx"
COCO_ANNOTATIONS_PATH = "app/_annotations.coco.json"
INPUT_SIZE = (560, 560)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 檢測設定
CONFIDENCE_THRESHOLD = 0.5
TOP_K = 30
NMS_THRESHOLD = 0.5

# 標註設定
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
]
FONT_SIZE = 20
BOX_THICKNESS = 3

# 速率限制
RATE_LIMIT = "10/minute"

# CORS 設定
ALLOW_ORIGINS = ["*"]
ALLOW_METHODS = ["GET", "POST"]
ALLOW_HEADERS = ["*"]

# 伺服器設定
HOST = "0.0.0.0"
PORT = 8000