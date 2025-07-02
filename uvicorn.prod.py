"""
生產環境 uvicorn 配置
統一使用 config.py 的設定，支援環境變數覆蓋
"""
import os
import uvicorn
from app.config import HOST, PORT

if __name__ == "__main__":
    # 環境變數優先，回退到 config.py 設定
    host = os.getenv("HOST", HOST)
    port = int(os.getenv("PORT", PORT))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        # 生產環境不啟用 reload
        reload=False
    )