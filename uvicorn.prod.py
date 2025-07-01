"""
生產環境 uvicorn 配置 - 輕量化推理優化
"""
import os
import uvicorn

if __name__ == "__main__":
    # 推理環境配置 (2核心+2G RAM)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        # 異步事件循環優化
        loop="uvloop",
        # HTTP 協議優化
        http="httptools",
        # 日誌設定
        access_log=True,
        log_level="info",
        # 性能優化
        backlog=2048,
    )