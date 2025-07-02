"""
藥丸檢測 API - 主應用
使用 Pillow 預處理流程，基於 RF-DETR ONNX 模型
OpenCV 依賴已移除，使用 numpy + Pillow 實現
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, HttpUrl

from app.config import *
from app.detection_service import DetectionService
from app import __version__

# 設置日誌
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 速率限制器
limiter = Limiter(key_func=get_remote_address)

# 回應模型
class DetectionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 啟動時初始化
    logger.info("🚀 正在初始化檢測服務...")
    try:
        detection_service = DetectionService()
        await detection_service.initialize()
        app.state.detection_service = detection_service
        logger.info("✅ 檢測服務初始化完成")
    except Exception as e:
        logger.error(f"❌ 檢測服務初始化失敗: {e}")
        raise
        
    yield
    
    # 關閉時清理
    logger.info("🔄 正在關閉檢測服務...")

# 創建 FastAPI 應用
app = FastAPI(
    title=API_TITLE,
    version=__version__,
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
async def root(request: Request):
    """API 根路徑"""
    detection_service = request.app.state.detection_service
    return {
        "message": "藥丸檢測 API",
        "version": __version__,
        "status": "running",
        "supported_classes": detection_service.get_classes() if detection_service and detection_service.is_ready() else [],
        "endpoints": ["/health", "/detect", "/test"]
    }

@app.get("/health")
async def health_check(request: Request):
    """健康檢查端點"""
    detection_service = request.app.state.detection_service
    is_ready = detection_service and detection_service.is_ready()
    
    return {
        "status": "healthy" if is_ready else "initializing",
        "service_ready": is_ready,
        "service_info": detection_service.get_service_info() if detection_service else None
    }

@app.post("/detect", response_model=DetectionResponse)
@limiter.limit(RATE_LIMIT_DETECTION)
async def detect_pills(
    request: Request, 
    image_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """統一檢測端點 - 支援 URL 和檔案上傳"""
    detection_service = request.app.state.detection_service
    if not detection_service or not detection_service.is_ready():
        raise HTTPException(status_code=503, detail="檢測服務尚未就緒")
    
    # 驗證輸入參數
    if not image_url and not file:
        raise HTTPException(status_code=400, detail="請提供 image_url 或上傳檔案")
    
    if image_url and file:
        raise HTTPException(status_code=400, detail="請只提供 image_url 或檔案，不可同時提供")
    
    try:
        if image_url:
            # 處理 URL 檢測
            result = await detection_service.detect_from_url(image_url)
            
        else:
            # 處理檔案上傳檢測
            if not file.filename:
                raise HTTPException(status_code=400, detail="請選擇一個檔案")
                
            file_ext = f".{file.filename.split('.')[-1].lower()}"
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"不支援的檔案格式。支援格式: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            file_content = await file.read()
            
            if len(file_content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"檔案太大，最大允許 {MAX_FILE_SIZE//1024//1024}MB")
            
            result = await detection_service.detect_from_file(file_content, file.filename)
        
        logger.info(f"✅ 檢測完成，發現 {result['total_detections']} 個藥丸")
        
        return DetectionResponse(
            success=True,
            message=f"檢測完成，發現 {result['total_detections']} 個藥丸",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 檢測失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="檢測服務暫時不可用，請稍後再試")

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
                <p>藥丸檢測測試介面</p>
            </div>
            
            <div class="section">
                <h3>📷 方法一：上傳圖片檔案</h3>
                <form id="fileForm">
                    <div class="form-group">
                        <input type="file" id="imageFile" accept="image/*" required>
                        <div id="imagePreview" style="margin-top: 10px;"></div>
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
                    
                    // 添加 JSON 結果複製功能
                    html += `<div style="margin-top: 15px;">
                        <button onclick="copyResult('json')" style="background: #28a745; margin-right: 10px;">📋 複製 JSON</button>
                        <button onclick="copyResult('detections')" style="background: #17a2b8;">📋 複製檢測結果</button>
                    </div>`;
                    
                    // 儲存當前結果供複製使用
                    window.currentResult = data;
                    
                    if (data.annotated_image) {{
                        html += `<img src="${{data.annotated_image}}" alt="檢測結果圖">`;
                    }}
                }}
                
                html += '</div>';
                resultDiv.innerHTML = html;
            }}
            
            // 圖片預覽功能
            document.getElementById('imageFile').addEventListener('change', function(e) {{
                const file = e.target.files[0];
                const preview = document.getElementById('imagePreview');
                
                if (file) {{
                    const reader = new FileReader();
                    reader.onload = function(e) {{
                        preview.innerHTML = `<img src="${{e.target.result}}" style="max-width: 300px; max-height: 200px; border: 1px solid #ddd; border-radius: 4px;" alt="預覽">`;
                    }};
                    reader.readAsDataURL(file);
                }} else {{
                    preview.innerHTML = '';
                }}
            }});
            
            // 複製結果功能
            function copyResult(type) {{
                if (!window.currentResult) return;
                
                let text;
                if (type === 'json') {{
                    text = JSON.stringify(window.currentResult, null, 2);
                }} else if (type === 'detections') {{
                    text = JSON.stringify(window.currentResult.detections, null, 2);
                }}
                
                navigator.clipboard.writeText(text).then(() => {{
                    alert('已複製到剪貼板！');
                }}).catch(err => {{
                    console.error('複製失敗:', err);
                    alert('複製失敗，請手動複製');
                }});
            }}
            
            // 檔案上傳表單
            document.getElementById('fileForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                showLoading();
                
                const fileInput = document.getElementById('imageFile');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {{
                    const response = await fetch('/detect', {{
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
                const formData = new FormData();
                formData.append('image_url', imageUrl);
                
                try {{
                    const response = await fetch('/detect', {{
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
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST, 
        port=PORT, 
        reload=RELOAD
    )