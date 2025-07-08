"""
藥丸檢測 API 全域設定檔
技術架構：resize → to_tensor → normalize
依賴：numpy + Pillow + ONNX Runtime（移除 OpenCV）
"""
import logging

# API 服務基本設定
API_VERSION = "2.1.4"
API_TITLE = "💊 藥丸檢測 API"
API_DESCRIPTION = "AI 藥丸識別服務 - 使用 RF-DETR ONNX 模型進行藥丸檢測"

# 上傳檔案相關設定
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# 模型與標註檔案設定
MODEL_PATH = "models/inference_model.onnx"
COCO_ANNOTATIONS_PATH = "app/_annotations.coco.json"
INPUT_SIZE = (560, 560)  # (width, height) - ONNX 模型輸入尺寸

# 圖像預處理設定（ImageNet 標準）
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB 通道均值
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB 通道標準差

# 檢測參數設定
CONFIDENCE_THRESHOLD = 0.5  # 信心度閾值
TOP_K = 30                  # 選擇前 K 個檢測結果
NMS_IOU_THRESHOLD = 0.5     # NMS IoU 閾值，用於移除重複檢測

# 圖像標註顯示設定
COLORS = [
    (220, 53, 69),   # 深紅色
    (40, 167, 69),   # 深綠色  
    (0, 123, 255),   # 藍色
    (255, 193, 7),   # 深黃色（更柔和）
    (111, 66, 193),  # 深紫色
    (23, 162, 184),  # 深青色
    (255, 109, 0),   # 深橙色
    (108, 117, 125), # 灰色
    (220, 20, 60),   # 深粉紅
    (34, 139, 34),   # 森林綠
]
FONT_SIZE = 16         # 標籤字體大小
BOX_THICKNESS = 3      # 邊界框線條粗細

# API 速率限制設定
RATE_LIMIT_GENERAL = "120/minute"    # 一般端點限制
RATE_LIMIT_DETECTION = "10/minute"   # 檢測端點限制（資源密集）
RATE_LIMIT_HEALTH = "60/minute"      # 健康檢查端點限制

# 跨來源請求（CORS）設定
ALLOW_ORIGINS = ["*"]  # 開發環境允許所有來源，生產環境應指定具體域名
ALLOW_CREDENTIALS = True
ALLOW_METHODS = ["GET", "POST"]
ALLOW_HEADERS = ["*"]

# 伺服器啟動參數
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True

# 日誌設定
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 性能監控設定
ENABLE_PERFORMANCE_LOGGING = True  # 啟用性能監控日誌
MAX_PROCESSING_TIME = 30.0         # 最大處理時間（秒）
MEMORY_LIMIT_MB = 1024             # 記憶體使用上限（MB）

# 模型推理設定
ONNX_PROVIDERS = ['CPUExecutionProvider']  # ONNX 執行提供者

# ONNX Runtime 性能優化設定
INTER_OP_NUM_THREADS = 0                   # 0=自動檢測CPU核心數進行跨節點並行
INTRA_OP_NUM_THREADS = 0                   # 0=自動檢測CPU核心數進行節點內並行
GRAPH_OPTIMIZATION_LEVEL = "ORT_ENABLE_BASIC"  # 圖優化級別：基本優化（Cloud Run 快速啟動）
EXECUTION_MODE = "ORT_PARALLEL"            # 執行模式：並行執行
ENABLE_MEM_PATTERN = False                 # 關閉內存模式優化（減少啟動時間）
ENABLE_CPU_MEM_ARENA = True                # 啟用CPU內存池（推理時有效）
ENABLE_PROFILING = False                   # 啟用性能分析（開發/調試用）

# 錯誤處理設定
MAX_RETRY_ATTEMPTS = 3             # 最大重試次數
RETRY_DELAY_SECONDS = 1.0          # 重試延遲時間

# 請求超時設定
REQUEST_TIMEOUT = 30               # HTTP請求超時（秒）
DETECTION_TIMEOUT = 60             # 檢測處理超時（秒）

# 圖像輸出設定
OUTPUT_IMAGE_FORMAT = "JPEG"       # 輸出圖像格式
OUTPUT_IMAGE_QUALITY = 80          # 輸出圖像品質 (1-100)

# 中文藥名映射表
CHINESE_DRUG_NAMES = {
    "Acetal": "愛舒疼錠",
    "Alprazolam": "安邦錠", 
    "Amoxicillin": "安莫西林膠囊",
    "Biotase": "妙化錠",
    "Bonstan": "普疏痛錠",
    "Cinnazine": "賜腦清錠",
    "Diovan_160mg": "得安穩錠160mg",
    "Diovan_80mg": "得安穩錠80mg",
    "Diphenidol": "敵芬尼朵錠",
    "Folacin": "葉酸錠",
    "Lansoprazole": "胃全膠囊",
    "Mozapry": "胃默適錠",
    "Nuspas": "痙得寧錠",
    "Relecox": "禮痛保膠囊",
    "Silence": "悠然錠",
    "Takepron": "泰克胃通錠",
    "Through": "便通樂錠",
    "Utraphen": "立除痛錠",
}

# 藥品代碼對應表
DRUG_CODES = {
    "Acetal": "A032320100",
    "Alprazolam": "AB577401G0", 
    "Amoxicillin": "A025866100",
    "Biotase": "A012876100",
    "Bonstan": "AC35705100",
    "Cinnazine": "AB01673100",
    "Diovan_160mg": "BC23374100",
    "Diovan_80mg": "BC23373100",
    "Diphenidol": "AC15078100",
    "Folacin": "AC346701G0",
    "Lansoprazole": "A044261100",
    "Mozapry": "AC55584100",
    "Nuspas": "AC48150100",
    "Relecox": "AC59295100",
    "Silence": "AC192471G0",
    "Takepron": "BC24273100",
    "Through": "A037697100",
    "Utraphen": "AC57777100",
}