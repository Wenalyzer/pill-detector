"""
藥丸檢測 API - 主應用
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, HttpUrl

from app.config import *
from app.pill_detector import PillDetector

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全域檢測器實例
detector: Optional[PillDetector] = None

# 速率限制器
limiter = Limiter(key_func=get_remote_address)

# 請求模型
class DetectionRequest(BaseModel):
    image_url: HttpUrl
    
class DetectionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    global detector
    
    # 啟動時初始化
    logger.info("🚀 正在初始化藥丸檢測器...")
    try:
        detector = PillDetector()
        await detector.initialize()
        logger.info("✅ 藥丸檢測器初始化完成")
    except Exception as e:
        logger.error(f"❌ 檢測器初始化失敗: {e}")
        raise
        
    yield
    
    # 關閉時清理
    logger.info("🔄 正在關閉檢測器...")

# 創建 FastAPI 應用
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS,
)

# 添加速率限制中間件
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.get("/")
async def root():
    """API 根路徑 - 顯示基本資訊"""
    return {
        "message": "歡迎使用藥丸檢測 API",
        "version": API_VERSION,
        "status": "running",
        "supported_classes": detector.get_classes() if detector else [],
        "endpoints": {
            "health": "/health",
            "classes": "/classes", 
            "detect": "/detect",
            "detect-file": "/detect-file",
            "test": "/test"
        }
    }

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    is_ready = detector and detector.is_ready()
    
    return {
        "status": "healthy" if is_ready else "initializing",
        "model_loaded": is_ready,
        "supported_classes": detector.get_classes() if detector else []
    }

@app.get("/classes")
async def get_classes():
    """獲取所有支援的藥丸類別"""
    if not detector:
        raise HTTPException(status_code=503, detail="檢測器尚未初始化")
        
    return {
        "classes": detector.get_classes(),
        "total": len(detector.get_classes())
    }

@app.post("/detect", response_model=DetectionResponse)
@limiter.limit(RATE_LIMIT)
async def detect_pills(request: Request, detection_request: DetectionRequest):
    """從圖像 URL 檢測藥丸"""
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="檢測器尚未就緒")
        
    try:
        logger.info(f"🔍 開始檢測圖像: {detection_request.image_url}")
        
        result = await detector.detect_from_url(str(detection_request.image_url))
        
        logger.info(f"✅ 檢測完成，發現 {result['total_detections']} 個藥丸")
        
        return DetectionResponse(
            success=True,
            message=f"檢測完成，發現 {result['total_detections']} 個藥丸",
            data=result
        )
        
    except Exception as e:
        logger.error(f"❌ 檢測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")

@app.post("/detect-file", response_model=DetectionResponse)
@limiter.limit(RATE_LIMIT)
async def detect_pills_from_file(request: Request, file: UploadFile = File(...)):
    """從上傳檔案檢測藥丸"""
    if not detector or not detector.is_ready():
        raise HTTPException(status_code=503, detail="檢測器尚未就緒")
        
    # 檔案驗證
    if not file.filename:
        raise HTTPException(status_code=400, detail="請選擇一個檔案")
        
    file_ext = f".{file.filename.split('.')[-1].lower()}"
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"不支援的檔案格式。支援格式: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # 讀取檔案內容
        file_content = await file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"檔案太大，最大允許 {MAX_FILE_SIZE//1024//1024}MB")
        
        logger.info(f"🔍 開始檢測上傳檔案: {file.filename}")
        
        result = await detector.detect_from_file(file_content)
        
        logger.info(f"✅ 檢測完成，發現 {result['total_detections']} 個藥丸")
        
        return DetectionResponse(
            success=True,
            message=f"檢測完成，發現 {result['total_detections']} 個藥丸",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 檔案檢測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Web 測試介面"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{API_TITLE} - 測試介面</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; color: #333; }}
            .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .form-group {{ margin-bottom: 15px; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #555; }}
            input, select {{ width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; }}
            button {{ background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }}
            button:hover {{ background: #0056b3; }}
            .result {{ margin-top: 20px; padding: 15px; border-radius: 4px; }}
            .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            .detection-list {{ margin-top: 10px; }}
            .detection-item {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 4px; border-left: 4px solid #007bff; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
            .loading {{ display: none; text-align: center; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 {API_TITLE}</h1>
                <p>上傳圖片或輸入圖片 URL 來檢測藥丸</p>
            </div>
            
            <div class="section">
                <h3>📷 方法一：上傳圖片檔案</h3>
                <form id="fileForm">
                    <div class="form-group">
                        <input type="file" id="imageFile" accept="image/*" required>
                    </div>
                    <button type="submit">🔍 開始檢測</button>
                </form>
            </div>
            
            <div class="section">
                <h3>🌐 方法二：圖片 URL</h3>
                <form id="urlForm">
                    <div class="form-group">
                        <input type="url" id="imageUrl" placeholder="請輸入圖片 URL" required>
                    </div>
                    <button type="submit">🔍 開始檢測</button>
                </form>
            </div>
            
            <div class="loading" id="loading">
                <p>⏳ 正在處理圖片，請稍候...</p>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            function showLoading() {{
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';
            }}
            
            function hideLoading() {{
                document.getElementById('loading').style.display = 'none';
            }}
            
            function showResult(success, message, data) {{
                const resultDiv = document.getElementById('result');
                const className = success ? 'success' : 'error';
                
                let html = `<div class="result ${{className}}"><h4>${{message}}</h4>`;
                
                if (success && data && data.detections) {{
                    html += `<p><strong>檢測到 ${{data.total_detections}} 個藥丸：</strong></p>`;
                    html += '<div class="detection-list">';
                    
                    data.detections.forEach((det, idx) => {{
                        html += `<div class="detection-item">
                            <strong>${{det.class_name}}</strong> - 信心度: ${{(det.confidence * 100).toFixed(1)}}%
                            <br>位置: [${{det.bbox.join(', ')}}]
                        </div>`;
                    }});
                    
                    html += '</div>';
                    
                    if (data.annotated_image) {{
                        html += `<img src="${{data.annotated_image}}" alt="檢測結果圖">`;
                    }}
                }}
                
                html += '</div>';
                resultDiv.innerHTML = html;
            }}
            
            // 檔案上傳表單
            document.getElementById('fileForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                showLoading();
                
                const fileInput = document.getElementById('imageFile');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {{
                    const response = await fetch('/detect-file', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    showResult(result.success, result.message, result.data);
                }} catch (error) {{
                    showResult(false, '請求失敗: ' + error.message);
                }} finally {{
                    hideLoading();
                }}
            }});
            
            // URL 表單
            document.getElementById('urlForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                showLoading();
                
                const imageUrl = document.getElementById('imageUrl').value;
                
                try {{
                    const response = await fetch('/detect', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ image_url: imageUrl }})
                    }});
                    
                    const result = await response.json();
                    showResult(result.success, result.message, result.data);
                }} catch (error) {{
                    showResult(false, '請求失敗: ' + error.message);
                }} finally {{
                    hideLoading();
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)