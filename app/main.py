from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, HttpUrl, Field
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import time
import logging
import json
import os
from contextlib import asynccontextmanager
import io
import requests
from typing import List, Dict, Optional

# 🔧 設置更詳細的日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全域變數
onnx_session = None
class_names = None

# 🎯 配置常數
class Config:
    MODEL_PATH = "./models/inference_model.onnx"
    ANNOTATIONS_PATH = "./app/_annotations.coco.json"
    INPUT_SIZE = (560, 560)
    DEFAULT_THRESHOLD = 0.5
    NUM_SELECT = 30
    
    # ImageNet 正規化參數
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # 檢測框顏色調色板 (BGR 格式)
    COLORS = [
        (0, 255, 0),      # 亮綠色
        (255, 0, 0),      # 亮藍色  
        (0, 0, 255),      # 亮紅色
        (0, 255, 255),    # 亮黃色
        (255, 0, 255),    # 亮紫色
        (255, 165, 0),    # 橙色
        (0, 128, 255),    # 淺藍色
        (128, 0, 128),    # 紫色
        (255, 192, 203),  # 粉紅色
        (0, 255, 127),    # 春綠色
    ]

def load_class_names_from_coco() -> Optional[List[str]]:
    """從 COCO annotations 文件載入類別名稱"""
    try:
        if not os.path.exists(Config.ANNOTATIONS_PATH):
            logger.error(f"❌ COCO 標註文件不存在: {Config.ANNOTATIONS_PATH}")
            return None
            
        with open(Config.ANNOTATIONS_PATH, "r", encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 按照 category_id 排序，確保索引正確
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        class_names = [cat['name'] for cat in categories]
        
        logger.info(f"✅ 成功載入 {len(class_names)} 個類別")
        logger.info(f"📋 類別列表: {class_names}")
        
        return class_names
        
    except Exception as e:
        logger.error(f"❌ 載入 COCO 標註文件失敗: {e}")
        return None

def preprocess_image_for_onnx(image_array: np.ndarray, input_size: tuple = Config.INPUT_SIZE) -> np.ndarray:
    """
    為 ONNX 模型預處理圖像 - 完全匹配 PyTorch 處理流程
    
    Args:
        image_array: RGB 格式的圖像數組
        input_size: 目標輸入尺寸 (width, height)
    
    Returns:
        預處理後的張量 (1, 3, H, W)
    """
    try:
        # 步驟1: F.to_tensor() - 轉換為 CHW 並正規化到 [0,1]
        tensor_like = image_array.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        # 步驟2: F.normalize() - ImageNet 正規化
        means = np.array(Config.IMAGENET_MEAN, dtype=np.float32).reshape(-1, 1, 1)
        stds = np.array(Config.IMAGENET_STD, dtype=np.float32).reshape(-1, 1, 1)
        normalized = (tensor_like - means) / stds
        
        # 步驟3: F.resize() - 調整到模型輸入尺寸
        hwc_normalized = normalized.transpose((1, 2, 0))  # CHW -> HWC for OpenCV
        resized_hwc = cv2.resize(hwc_normalized, input_size, interpolation=cv2.INTER_LINEAR)
        resized_chw = resized_hwc.transpose((2, 0, 1))  # HWC -> CHW
        
        # 步驟4: 添加 batch 維度
        batched = np.expand_dims(resized_chw, axis=0)
        
        logger.debug(f"🔧 預處理統計: 範圍[{np.min(batched):.3f}, {np.max(batched):.3f}], 均值{np.mean(batched):.3f}")
        
        return batched
        
    except Exception as e:
        logger.error(f"❌ 圖像預處理失敗: {e}")
        raise

def postprocess_onnx_output(
    outputs: List[np.ndarray], 
    threshold: float = Config.DEFAULT_THRESHOLD,
    target_size: tuple = Config.INPUT_SIZE,
    num_select: int = Config.NUM_SELECT
) -> Dict[str, np.ndarray]:
    """
    後處理 ONNX 模型輸出 - RF-DETR 後處理邏輯
    
    Args:
        outputs: ONNX 模型輸出 [boxes, logits]
        threshold: 信心度閾值
        target_size: 目標圖像尺寸 (width, height)
        num_select: Top-K 選擇數量
    
    Returns:
        檢測結果字典
    """
    try:
        pred_boxes, pred_logits = outputs[0][0], outputs[1][0]  # 移除 batch 維度
        
        # Sigmoid 激活
        prob = 1.0 / (1.0 + np.exp(-pred_logits))
        
        # Top-K 選擇
        prob_flat = prob.reshape(-1)
        topk_indices = np.argpartition(prob_flat, -num_select)[-num_select:]
        topk_indices = topk_indices[np.argsort(prob_flat[topk_indices])[::-1]]
        
        scores = prob_flat[topk_indices]
        topk_boxes = topk_indices // pred_logits.shape[1]
        labels = topk_indices % pred_logits.shape[1]
        
        # 邊界框格式轉換 cxcywh -> xyxy
        def box_cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1, y1 = cx - 0.5 * w, cy - 0.5 * h
            x2, y2 = cx + 0.5 * w, cy + 0.5 * h
            return np.stack([x1, y1, x2, y2], axis=1)
        
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        selected_boxes = boxes_xyxy[topk_boxes]
        
        # 縮放到目標尺寸
        img_w, img_h = target_size
        scale_fct = np.array([img_w, img_h, img_w, img_h])
        final_boxes = selected_boxes * scale_fct
        
        # 應用閾值過濾
        valid_mask = scores >= threshold
        if not np.any(valid_mask):
            logger.info("ℹ️ 沒有檢測結果超過閾值")
            return {
                'xyxy': np.array([]).reshape(0, 4),
                'confidence': np.array([]),
                'class_id': np.array([])
            }
        
        filtered_boxes = final_boxes[valid_mask]
        filtered_scores = scores[valid_mask]
        filtered_labels = labels[valid_mask]
        
        # 座標範圍限制
        filtered_boxes[:, [0, 2]] = np.clip(filtered_boxes[:, [0, 2]], 0, img_w)
        filtered_boxes[:, [1, 3]] = np.clip(filtered_boxes[:, [1, 3]], 0, img_h)
        
        logger.info(f"✅ 檢測到 {len(filtered_boxes)} 個對象")
        
        return {
            'xyxy': filtered_boxes,
            'confidence': filtered_scores,
            'class_id': filtered_labels
        }
        
    except Exception as e:
        logger.error(f"❌ 後處理失敗: {e}")
        raise

def get_optimal_font(font_size: int) -> ImageFont.ImageFont:
    """獲取最佳可用字體"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except:
            continue
    
    return ImageFont.load_default()

def draw_high_quality_text(
    image_array: np.ndarray, 
    text: str, 
    position: tuple, 
    font_size: int, 
    bg_color: tuple
) -> np.ndarray:
    """使用 PIL 繪製高質量文字"""
    try:
        # 轉換到 PIL
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 獲取字體
        font = get_optimal_font(font_size)
        
        # 計算文字尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        x, y = position
        padding = 4
        
        # 繪製背景
        bg_bbox = [
            x - padding,
            y - text_height - padding,
            x + text_width + padding,
            y + padding
        ]
        
        bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])  # BGR -> RGB
        draw.rectangle(bg_bbox, fill=bg_color_rgb)
        
        # 繪製文字
        draw.text((x, y - text_height), text, font=font, fill=(255, 255, 255))
        
        # 轉換回 OpenCV 格式
        result_rgb = np.array(pil_image)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"❌ 文字繪製失敗: {e}")
        return image_array

def calculate_smart_label_positions(
    detections: Dict[str, np.ndarray], 
    image_size: tuple, 
    font_size: int
) -> List[tuple]:
    """智能計算標籤位置，避免重疊和遮擋檢測框"""
    w, h = image_size
    boxes = detections['xyxy']
    confidences = detections['confidence']
    
    if len(boxes) == 0:
        return []
    
    label_height = font_size + 8
    max_label_width = font_size * 12
    
    label_positions = []
    occupied_regions = []
    
    # 按信心度排序
    sorted_indices = np.argsort(confidences)[::-1]
    
    for idx in sorted_indices:
        box = boxes[idx]
        class_id = detections['class_id'][idx]
        conf = confidences[idx]
        
        pill_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
        label_text = f"{pill_name} {conf:.2f}"
        label_width = min(len(label_text) * font_size * 0.6, max_label_width)
        
        x1, y1, x2, y2 = box.astype(int)
        
        # 候選位置 - 增加更多選項
        candidates = [
            # 上方 - 多個位置
            (x1, y1 - label_height - 5, x1 + label_width, y1 - 5),
            (x1 + (x2-x1)//4, y1 - label_height - 5, x1 + (x2-x1)//4 + label_width, y1 - 5),
            (max(0, x2 - label_width), y1 - label_height - 5, x2, y1 - 5),
            
            # 下方 - 多個位置  
            (x1, y2 + 5, x1 + label_width, y2 + 5 + label_height),
            (x1 + (x2-x1)//4, y2 + 5, x1 + (x2-x1)//4 + label_width, y2 + 5 + label_height),
            (max(0, x2 - label_width), y2 + 5, x2, y2 + 5 + label_height),
            
            # 左側
            (x1 - label_width - 5, y1, x1 - 5, y1 + label_height),
            (x1 - label_width - 5, y1 + (y2-y1)//4, x1 - 5, y1 + (y2-y1)//4 + label_height),
            
            # 右側
            (x2 + 5, y1, x2 + 5 + label_width, y1 + label_height),
            (x2 + 5, y1 + (y2-y1)//4, x2 + 5 + label_width, y1 + (y2-y1)//4 + label_height),
            
            # 框內上方
            (x1 + 5, y1 + 5, x1 + 5 + label_width, y1 + 5 + label_height),
            # 框內下方
            (x1 + 5, y2 - label_height - 5, x1 + 5 + label_width, y2 - 5),
        ]
        
        # 找最佳位置
        best_position = None
        for pos_x1, pos_y1, pos_x2, pos_y2 in candidates:
            # 1. 邊界檢查
            if pos_x1 < 0 or pos_y1 < 0 or pos_x2 > w or pos_y2 > h:
                continue
            
            # 2. 與其他標籤重疊檢查
            overlaps_label = False
            for occupied in occupied_regions:
                if not (pos_x2 < occupied[0] or pos_x1 > occupied[2] or 
                       pos_y2 < occupied[1] or pos_y1 > occupied[3]):
                    overlaps_label = True
                    break
            
            if overlaps_label:
                continue
            
            # 3. 🔍 檢查是否遮擋其他檢測框 (這是關鍵！)
            blocks_other_boxes = False
            for other_idx in range(len(boxes)):
                if other_idx == idx:  # 跳過自己
                    continue
                    
                other_box = boxes[other_idx]
                ox1, oy1, ox2, oy2 = other_box.astype(int)
                
                # 計算重疊面積
                overlap_x1 = max(pos_x1, ox1)
                overlap_y1 = max(pos_y1, oy1)
                overlap_x2 = min(pos_x2, ox2)
                overlap_y2 = min(pos_y2, oy2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    box_area = (ox2 - ox1) * (oy2 - oy1)
                    
                    # 🎯 如果標籤遮擋其他框超過 20%，則不合適
                    if overlap_area > box_area * 0.2:  # 從 30% 降到 20% 更嚴格
                        blocks_other_boxes = True
                        break
            
            if not blocks_other_boxes:
                best_position = (pos_x1, pos_y1, pos_x2, pos_y2)
                break
        
        # 4. 🚨 備援位置 - 如果都不行，尋找圖像邊緣空白區域
        if best_position is None:
            # 嘗試圖像四個角落
            corner_candidates = [
                # 左上角
                (5, 5, 5 + label_width, 5 + label_height),
                # 右上角  
                (w - label_width - 5, 5, w - 5, 5 + label_height),
                # 左下角
                (5, h - label_height - 5, 5 + label_width, h - 5),
                # 右下角
                (w - label_width - 5, h - label_height - 5, w - 5, h - 5),
            ]
            
            for pos_x1, pos_y1, pos_x2, pos_y2 in corner_candidates:
                # 檢查角落位置是否與已有標籤重疊
                corner_overlap = False
                for occupied in occupied_regions:
                    if not (pos_x2 < occupied[0] or pos_x1 > occupied[2] or 
                           pos_y2 < occupied[1] or pos_y1 > occupied[3]):
                        corner_overlap = True
                        break
                
                if not corner_overlap:
                    best_position = (pos_x1, pos_y1, pos_x2, pos_y2)
                    break
            
            # 5. 🔧 最終備援：強制放在框上方（即使可能重疊）
            if best_position is None:
                pos_x1 = max(5, min(x1, w - label_width - 5))
                pos_y1 = max(label_height + 5, y1 - 10)
                best_position = (pos_x1, pos_y1, pos_x1 + label_width, pos_y1 + label_height)
        
        # 記錄位置
        label_positions.append((best_position[0], best_position[1] + label_height))
        occupied_regions.append(best_position)
    
    # 重新排序到原始順序
    final_positions = [None] * len(boxes)
    for i, idx in enumerate(sorted_indices):
        final_positions[idx] = label_positions[i]
    
    return final_positions

def save_image_to_base64(image: Image.Image) -> str:
    """將圖片轉為 base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    import base64
    return base64.b64encode(buffer.getvalue()).decode()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    global onnx_session, class_names
    
    try:
        logger.info("🚀 啟動藥丸檢測 API...")
        
        # 載入類別名稱
        class_names = load_class_names_from_coco()
        if class_names is None:
            raise RuntimeError("無法載入類別名稱")
        
        # 載入 ONNX 模型
        if not os.path.exists(Config.MODEL_PATH):
            raise RuntimeError(f"模型文件不存在: {Config.MODEL_PATH}")
        
        logger.info(f"📦 載入 ONNX 模型: {Config.MODEL_PATH}")
        onnx_session = ort.InferenceSession(
            Config.MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        
        # 模型信息
        input_info = onnx_session.get_inputs()[0]
        output_info = onnx_session.get_outputs()
        
        logger.info(f"✅ 模型載入成功")
        logger.info(f"📥 輸入: {input_info.name}{input_info.shape}")
        logger.info(f"📤 輸出: {[out.name + str(out.shape) for out in output_info]}")
        logger.info(f"🎯 支援 {len(class_names)} 種藥丸")
        logger.info("🌟 API 準備就緒！")
        
    except Exception as e:
        logger.error(f"❌ 初始化失敗: {e}")
        raise
    
    yield
    
    logger.info("🛑 關閉藥丸檢測 API")
    onnx_session = None
    class_names = None

# 🚦 創建速率限制器
limiter = Limiter(key_func=get_remote_address)

# FastAPI 應用
app = FastAPI(
    title="💊 Pill Detection API",
    description="AI 藥丸識別服務 - 使用 RF-DETR 模型進行高精度藥丸檢測",
    version="2.1.0",
    lifespan=lifespan
)

# 🚦 設置限流器和異常處理
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 請求/回應模型
class DetectionRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="圖像 URL")
    threshold: float = Field(
        default=Config.DEFAULT_THRESHOLD, 
        ge=0.1, le=1.0, 
        description="檢測信心度閾值 (0.1-1.0)"
    )

class DetectionResult(BaseModel):
    detection_id: int
    class_id: int
    pill_name: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    success: bool
    detections: List[DetectionResult]
    annotated_image_base64: str
    inference_time_ms: float
    total_detections: int

# API 端點
@app.get("/", tags=["系統"])
@limiter.limit("120/minute")
async def root(request: Request):
    """API 根端點"""
    return {
        "name": "💊 Pill Detection API",
        "version": "2.1.0",
        "status": "running",
        "model": "RF-DETR ONNX",
        "supported_pills": len(class_names) if class_names else 0
    }

@app.get("/health", tags=["系統"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """健康檢查"""
    return {
        "status": "healthy",
        "model_loaded": onnx_session is not None,
        "classes_loaded": class_names is not None,
        "total_classes": len(class_names) if class_names else 0,
        "memory_usage": "OK",
        "timestamp": time.time()
    }

@app.get("/classes", tags=["模型"])
@limiter.limit("30/minute")
async def get_supported_classes(request: Request):
    """獲取所有支援的藥丸類別"""
    if not class_names:
        raise HTTPException(status_code=503, detail="類別名稱未載入")
    
    return {
        "total_classes": len(class_names),
        "classes": [
            {"id": i, "name": name} 
            for i, name in enumerate(class_names)
        ]
    }

@app.post("/detect", response_model=DetectionResponse, tags=["檢測"])
@limiter.limit("10/minute")  # 🚦 每分鐘最多 10 次請求
async def detect_pills(request: Request, detection_request: DetectionRequest):
    """
    藥丸檢測主端點
    
    上傳圖像 URL，返回檢測到的藥丸信息和標註圖像
    """
    if not onnx_session or not class_names:
        raise HTTPException(status_code=503, detail="服務未就緒")
    
    try:
        # 🔽 下載圖像
        logger.info(f"📥 下載圖像: {detection_request.image_url}")
        response = requests.get(str(detection_request.image_url), timeout=15)
        response.raise_for_status()
        
        pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        orig_w, orig_h = pil_image.size
        logger.info(f"🖼️ 圖像尺寸: {orig_w}x{orig_h}")
        
        # 🔄 預處理
        image_array = np.array(pil_image)
        input_tensor = preprocess_image_for_onnx(image_array)
        
        # 🤖 推理
        start_time = time.time()
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: input_tensor})
        inference_time = (time.time() - start_time) * 1000
        
        # 🔍 後處理
        detections = postprocess_onnx_output(
            outputs, 
            threshold=detection_request.threshold,
            target_size=(orig_w, orig_h)
        )
        
        total_detections = len(detections['class_id'])
        logger.info(f"🎯 檢測到 {total_detections} 個藥丸")
        
        # 🎨 繪製結果 - 修正繪製順序
        annotated_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        if total_detections > 0:
            h, w = image_array.shape[:2]
            font_size = max(16, int(min(w, h) / 35))
            thickness = max(2, int(min(w, h) / 400))
            
            # 智能標籤位置
            label_positions = calculate_smart_label_positions(detections, (w, h), font_size)
            
            # 🎯 步驟1: 先繪製所有檢測框
            for i, (box, conf, class_id) in enumerate(zip(
                detections['xyxy'], 
                detections['confidence'], 
                detections['class_id']
            )):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                color = Config.COLORS[class_id % len(Config.COLORS)]
                
                # 只繪製邊界框，不繪製標籤
                cv2.rectangle(annotated_array, (x1, y1), (x2, y2), color, thickness)
            
            # 🏷️ 步驟2: 再繪製所有標籤 (這樣標籤會在框的上層)
            for i, (box, conf, class_id) in enumerate(zip(
                detections['xyxy'], 
                detections['confidence'], 
                detections['class_id']
            )):
                pill_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                color = Config.COLORS[class_id % len(Config.COLORS)]
                
                # 繪製標籤
                label = f"{pill_name} {conf:.2f}"
                text_x, text_y = label_positions[i]
                annotated_array = draw_high_quality_text(
                    annotated_array, label, (text_x, text_y), font_size, color
                )
        
        # 📤 輸出結果
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB))
        image_base64 = save_image_to_base64(annotated_pil)
        
        detection_results = [
            DetectionResult(
                detection_id=i + 1,
                class_id=int(class_id),
                pill_name=class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}",
                confidence=float(conf),
                bbox=[float(x) for x in box]
            )
            for i, (class_id, conf, box) in enumerate(zip(
                detections['class_id'], 
                detections['confidence'], 
                detections['xyxy']
            ))
        ]
        
        logger.info(f"✅ 處理完成 - 檢測:{total_detections}, 推理:{inference_time:.1f}ms")
        
        return DetectionResponse(
            success=True,
            detections=detection_results,
            annotated_image_base64=image_base64,
            inference_time_ms=round(inference_time, 2),
            total_detections=total_detections
        )
        
    except requests.RequestException as e:
        logger.error(f"❌ 圖像下載失敗: {e}")
        raise HTTPException(status_code=400, detail="無法下載圖像")
    except Exception as e:
        logger.error(f"❌ 檢測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"檢測處理失敗: {str(e)}")

@app.post("/detect-file", tags=["檢測"])
@limiter.limit("10/minute")
async def detect_pills_from_file(
    request: Request,
    file: UploadFile = File(..., description="上傳的圖片檔案"),
    threshold: float = Form(default=Config.DEFAULT_THRESHOLD, ge=0.1, le=1.0)
):
    """
    直接上傳圖片檔案進行藥丸檢測
    
    支援 JPG, PNG 等常見圖片格式
    """
    if not onnx_session or not class_names:
        raise HTTPException(status_code=503, detail="服務未就緒")
    
    try:
        # 檢查檔案類型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="請上傳圖片檔案")
        
        # 📥 讀取上傳的圖片
        logger.info(f"📥 處理上傳檔案: {file.filename}")
        image_data = await file.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        orig_w, orig_h = pil_image.size
        logger.info(f"🖼️ 圖像尺寸: {orig_w}x{orig_h}")
        
        # 🔄 預處理
        image_array = np.array(pil_image)
        input_tensor = preprocess_image_for_onnx(image_array)
        
        # 🤖 推理
        start_time = time.time()
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: input_tensor})
        inference_time = (time.time() - start_time) * 1000
        
        # 🔍 後處理
        detections = postprocess_onnx_output(
            outputs, 
            threshold=threshold,
            target_size=(orig_w, orig_h)
        )
        
        total_detections = len(detections['class_id'])
        logger.info(f"🎯 檢測到 {total_detections} 個藥丸")
        
        # 🎨 繪製結果 (使用相同的繪製邏輯)
        annotated_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        if total_detections > 0:
            h, w = image_array.shape[:2]
            font_size = max(16, int(min(w, h) / 35))
            thickness = max(2, int(min(w, h) / 400))
            
            # 智能標籤位置
            label_positions = calculate_smart_label_positions(detections, (w, h), font_size)
            
            # 先繪製所有檢測框
            for i, (box, conf, class_id) in enumerate(zip(
                detections['xyxy'], 
                detections['confidence'], 
                detections['class_id']
            )):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                color = Config.COLORS[class_id % len(Config.COLORS)]
                cv2.rectangle(annotated_array, (x1, y1), (x2, y2), color, thickness)
            
            # 再繪製所有標籤
            for i, (box, conf, class_id) in enumerate(zip(
                detections['xyxy'], 
                detections['confidence'], 
                detections['class_id']
            )):
                pill_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                color = Config.COLORS[class_id % len(Config.COLORS)]
                
                label = f"{pill_name} {conf:.2f}"
                text_x, text_y = label_positions[i]
                annotated_array = draw_high_quality_text(
                    annotated_array, label, (text_x, text_y), font_size, color
                )
        
        # 📤 輸出結果
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB))
        image_base64 = save_image_to_base64(annotated_pil)
        
        detection_results = [
            DetectionResult(
                detection_id=i + 1,
                class_id=int(class_id),
                pill_name=class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}",
                confidence=float(conf),
                bbox=[float(x) for x in box]
            )
            for i, (class_id, conf, box) in enumerate(zip(
                detections['class_id'], 
                detections['confidence'], 
                detections['xyxy']
            ))
        ]
        
        logger.info(f"✅ 檔案處理完成 - 檢測:{total_detections}, 推理:{inference_time:.1f}ms")
        
        return {
            "success": True,
            "filename": file.filename,
            "detections": detection_results,
            "annotated_image_base64": image_base64,
            "inference_time_ms": round(inference_time, 2),
            "total_detections": total_detections
        }
        
    except Exception as e:
        logger.error(f"❌ 檔案檢測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"檔案處理失敗: {str(e)}")

@app.get("/test", response_class=HTMLResponse, tags=["工具"])
async def test_page():
    """Web 測試界面 - 直接上傳圖片進行檢測"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>💊 藥丸偵測測試</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 { 
                color: #4a5568; 
                text-align: center; 
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #718096;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .form-group { 
                margin: 20px 0; 
            }
            label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: 600;
                color: #2d3748;
            }
            input[type="file"] { 
                width: 100%; 
                padding: 12px; 
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                background: #f7fafc;
                transition: all 0.3s;
            }
            input[type="file"]:hover {
                border-color: #667eea;
                background: #edf2f7;
            }
            input[type="number"] { 
                width: 200px; 
                padding: 10px; 
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                font-size: 16px;
            }
            button { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 12px 30px; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: transform 0.2s;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #a0aec0;
                cursor: not-allowed;
                transform: none;
            }
            #results { 
                margin-top: 30px; 
                padding: 20px; 
                border: 1px solid #e2e8f0; 
                border-radius: 10px;
                background: #f8fafc;
            }
            .detection { 
                background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px;
                border-left: 4px solid #38b2ac;
            }
            .detection strong {
                color: #2c7a7b;
                font-size: 1.1em;
            }
            img { 
                max-width: 100%; 
                height: auto; 
                margin-top: 15px; 
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .loading {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
                border-left: 4px solid #e53e3e;
                color: #742a2a;
            }
            .success-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .stat-number {
                font-size: 1.8em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #718096;
                font-size: 0.9em;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 藥丸偵測系統</h1>
            <p class="subtitle">上傳圖片，AI 立即識別藥丸種類</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label>📷 選擇圖片檔案:</label>
                    <input type="file" id="imageFile" accept="image/*" required>
                    <small style="color: #718096;">支援 JPG, PNG, JPEG 等格式</small>
                </div>
                
                <div class="form-group">
                    <label>🎯 信心度閾值:</label>
                    <input type="number" id="threshold" value="0.5" min="0.1" max="1.0" step="0.1">
                    <small style="color: #718096;">數值越高，檢測越嚴格 (建議 0.3-0.7)</small>
                </div>
                
                <div class="form-group">
                    <button type="submit" id="submitBtn">🚀 開始偵測</button>
                </div>
            </form>
            
            <div id="results" style="display:none;"></div>
        </div>
        
        <script>
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('imageFile');
            const thresholdInput = document.getElementById('threshold');
            const submitBtn = document.getElementById('submitBtn');
            
            if (!fileInput.files[0]) {
                alert('請先選擇圖片檔案！');
                return;
            }
            
            // 檢查檔案大小 (限制 10MB)
            if (fileInput.files[0].size > 10 * 1024 * 1024) {
                alert('檔案太大！請選擇小於 10MB 的圖片。');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('threshold', thresholdInput.value);
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            // 顯示載入狀態
            submitBtn.disabled = true;
            submitBtn.innerHTML = '🔄 偵測中...';
            resultsDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <span>AI 正在分析圖片，請稍候...</span>
                </div>
            `;
            
            try {
                const response = await fetch('/detect-file', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP ${response.status}`);
                }
                
                const result = await response.json();
                
                let html = '<h2>📊 偵測結果</h2>';
                
                // 統計卡片
                html += '<div class="success-stats">';
                html += `
                    <div class="stat-card">
                        <div class="stat-number">${result.total_detections}</div>
                        <div class="stat-label">偵測到的藥丸</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${result.inference_time_ms}ms</div>
                        <div class="stat-label">推論時間</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${(parseFloat(thresholdInput.value) * 100).toFixed(0)}%</div>
                        <div class="stat-label">信心度閾值</div>
                    </div>
                `;
                html += '</div>';
                
                html += `<p><strong>檔案名稱:</strong> ${result.filename}</p>`;
                
                if (result.total_detections > 0) {
                    html += '<h3>🔍 偵測詳情:</h3>';
                    result.detections.forEach((detection, index) => {
                        html += `<div class="detection">
                            <strong>${index + 1}. ${detection.pill_name}</strong><br>
                            <span style="font-size: 0.9em;">
                                🎯 信心度: ${(detection.confidence * 100).toFixed(1)}% | 
                                📍 位置: (${Math.round(detection.bbox[0])}, ${Math.round(detection.bbox[1])}) - 
                                (${Math.round(detection.bbox[2])}, ${Math.round(detection.bbox[3])})
                            </span>
                        </div>`;
                    });
                } else {
                    html += '<div class="detection" style="background: #fef5e7; border-left-color: #ed8936;">';
                    html += '<strong>🤷‍♂️ 未偵測到藥丸</strong><br>';
                    html += '<span style="font-size: 0.9em;">可以嘗試降低信心度閾值，或確認圖片中是否包含支援的藥丸類型。</span>';
                    html += '</div>';
                }
                
                if (result.annotated_image_base64) {
                    html += '<h3>📸 標註結果:</h3>';
                    html += `<img src="data:image/jpeg;base64,${result.annotated_image_base64}" alt="標註結果">`;
                }
                
                resultsDiv.innerHTML = html;
                
            } catch (error) {
                resultsDiv.innerHTML = `<div class="detection error">
                    <h3>❌ 偵測失敗</h3>
                    <p><strong>錯誤訊息:</strong> ${error.message}</p>
                    <p>請檢查：</p>
                    <ul>
                        <li>圖片格式是否正確 (JPG, PNG)</li>
                        <li>檔案大小是否小於 10MB</li>
                        <li>網路連線是否正常</li>
                    </ul>
                    <p>如問題持續，請稍後再試。</p>
                </div>`;
            } finally {
                // 恢復按鈕狀態
                submitBtn.disabled = false;
                submitBtn.innerHTML = '🚀 開始偵測';
            }
        }
        
        // 檔案選擇預覽
        document.getElementById('imageFile').onchange = function(e) {
            const file = e.target.files[0];
            if (file) {
                console.log(`選擇了檔案: ${file.name} (${(file.size/1024/1024).toFixed(2)}MB)`);
            }
        }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )