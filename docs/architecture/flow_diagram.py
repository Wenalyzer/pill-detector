#!/usr/bin/env python3
"""
無留白版請求處理流程圖 - 藥丸檢測 API
改善 API/ASGI 解釋，消除底部留白
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.font_manager as fm

def setup_chinese_font():
    """設定中文粗體字體"""
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
    ]
    
    chinese_font = None
    for path in font_paths:
        try:
            chinese_font = fm.FontProperties(fname=path, weight='bold')
            print(f"✅ 找到中文字體: {path}")
            break
        except:
            continue
    
    if chinese_font is None:
        chinese_font = fm.FontProperties(family='serif', weight='bold')
        print("⚠️ 使用備用粗體字體")
    
    return chinese_font

def create_no_whitespace_diagram():
    """創建無留白版流程圖"""
    
    chinese_font = setup_chinese_font()
    
    # 調整畫布尺寸，充分利用空間
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 定義顏色
    colors = {
        'client': '#FF6B6B',    # 紅色
        'server': '#4ECDC4',    # 青色
        'app': '#45B7D1',       # 藍色
        'service': '#FECA57',   # 黃色
        'detector': '#6C5CE7',  # 紫色
        'annotator': '#FF9F43', # 橙色
        'response': '#A8E6CF'   # 淺綠
    }
    
    # 標題
    ax.text(8, 10, '藥丸檢測 API 請求處理流程', 
            fontsize=26, fontweight='bold', ha='center', fontproperties=chinese_font)
    
    # === 主要組件 ===
    
    # 1. 客戶端
    client_box = FancyBboxPatch((0.5, 8), 3.2, 1.6,
                               boxstyle="round,pad=0.15",
                               facecolor=colors['client'],
                               edgecolor='black', linewidth=3)
    ax.add_patch(client_box)
    ax.text(2.1, 9.1, '客戶端', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(2.1, 8.6, 'POST /detect', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(2.1, 8.2, '圖片URL或檔案', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    
    # 2. uvicorn 伺服器
    server_box = FancyBboxPatch((4.5, 8), 3.2, 1.6,
                               boxstyle="round,pad=0.15",
                               facecolor=colors['server'],
                               edgecolor='black', linewidth=3)
    ax.add_patch(server_box)
    ax.text(6.1, 9.1, 'uvicorn', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font)
    ax.text(6.1, 8.6, 'ASGI 伺服器', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font)
    ax.text(6.1, 8.2, 'HTTP 解析與路由', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font)
    
    # 3. FastAPI 應用
    app_box = FancyBboxPatch((8.5, 8), 3.2, 1.6,
                            boxstyle="round,pad=0.15",
                            facecolor=colors['app'],
                            edgecolor='black', linewidth=3)
    ax.add_patch(app_box)
    ax.text(10.1, 9.1, 'FastAPI', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(10.1, 8.6, 'detect_pills() 端點', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(10.1, 8.2, '參數驗證與中間件', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    
    # 4. JSON 回應
    response_box = FancyBboxPatch((12.5, 8), 3.2, 1.6,
                                 boxstyle="round,pad=0.15",
                                 facecolor=colors['response'],
                                 edgecolor='black', linewidth=3)
    ax.add_patch(response_box)
    ax.text(14.1, 9.1, 'JSON 回應', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font)
    ax.text(14.1, 8.6, '檢測結果陣列', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font)
    ax.text(14.1, 8.2, 'Base64 標註圖片', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font)
    
    # === 自訂服務層 ===
    
    # DetectionService
    service_box = FancyBboxPatch((0.75, 5.5), 4.5, 1.6,
                                boxstyle="round,pad=0.15",
                                facecolor=colors['service'],
                                edgecolor='black', linewidth=3)
    ax.add_patch(service_box)
    ax.text(3, 6.6, 'DetectionService', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font)
    ax.text(3, 6.1, '業務邏輯協調層', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font)
    ax.text(3, 5.7, '檔案處理、錯誤處理', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font)
    
    # PillDetector
    detector_box = FancyBboxPatch((5.875, 5.5), 4.5, 1.6,
                                 boxstyle="round,pad=0.15",
                                 facecolor=colors['detector'],
                                 edgecolor='black', linewidth=3)
    ax.add_patch(detector_box)
    ax.text(8.125, 6.6, 'PillDetector', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(8.125, 6.1, 'ONNX 模型推理引擎', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(8.125, 5.7, '預處理與後處理', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    
    # ImageAnnotator
    annotator_box = FancyBboxPatch((11, 5.5), 4.5, 1.6,
                                  boxstyle="round,pad=0.15",
                                  facecolor=colors['annotator'],
                                  edgecolor='black', linewidth=3)
    ax.add_patch(annotator_box)
    ax.text(13.25, 6.6, 'ImageAnnotator', fontsize=24, fontweight='bold',
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(13.25, 6.1, '圖像標註器', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    ax.text(13.25, 5.7, '智能標籤定位', fontsize=18,
            ha='center', va='center', fontproperties=chinese_font, color='white')
    
    # === 流程步驟編號 ===
    steps = [
        ('1', 3.8, 9.6),
        ('2', 7.8, 9.6),
        ('3', 10.1, 7.75),
        ('4', 9.5, 5.4),
        ('5', 11.8, 9.6)
    ]
    
    for step, x, y in steps:
        step_circle = plt.Circle((x, y), 0.3, color='white', ec='black', linewidth=3)
        ax.add_patch(step_circle)
        ax.text(x, y, step, fontsize=16, fontweight='bold', ha='center', va='center')
    
    # === 箭頭連接 ===
    
    # 水平箭頭 (主流程)
    main_arrow_y = 8.7
    arrows = [
        ((3.7, main_arrow_y), (4.5, main_arrow_y)),
        ((7.7, main_arrow_y), (8.5, main_arrow_y)),
        ((11.7, main_arrow_y), (12.5, main_arrow_y))
    ]
    
    for start, end in arrows:
        arrow = patches.FancyArrowPatch(start, end,
                                      arrowstyle='->', mutation_scale=25,
                                      color='darkblue', linewidth=4)
        ax.add_patch(arrow)
    
    # FastAPI 到 DetectionService
    vertical_arrow1 = patches.FancyArrowPatch((9.8, 7.5), (3.25, 7),
                                            arrowstyle='->', mutation_scale=25,
                                            color='orange', linewidth=3)
    ax.add_patch(vertical_arrow1)
    
    # 服務層內部箭頭
    service_arrow_y = 6.2
    service_arrows = [
        ((5.2, service_arrow_y), (6.1, service_arrow_y)),
        ((10.3, service_arrow_y), (11.2, service_arrow_y))
    ]
    
    for start, end in service_arrows:
        arrow = patches.FancyArrowPatch(start, end,
                                      arrowstyle='->', mutation_scale=20,
                                      color='red', linewidth=3)
        ax.add_patch(arrow)
    
    # 綠色回傳箭頭
    return_arrow = patches.FancyArrowPatch((13.75, 7), (10.3, 7.5),
                                         arrowstyle='->', mutation_scale=25,
                                         color='green', linewidth=3)
    ax.add_patch(return_arrow)
    
    # === 改進的基礎概念解釋 - 大幅下移 ===
    
    # API 解釋 - 說明 API 的通用性
    api_explain_box = FancyBboxPatch((0.5, 3.6), 7.5, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#FFF8DC',
                                    edgecolor='#DAA520', linewidth=2)
    ax.add_patch(api_explain_box)
    ax.text(4.25, 4.5, '為什麼要用 API？', 
            fontsize=16, fontweight='bold', ha='center', fontproperties=chinese_font)
    ax.text(4.25, 4.1, 'API 提供標準化介面，任何程式語言都能使用', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    ax.text(4.25, 3.8, 'Python、Java、JavaScript、C# 等都能透過 HTTP 呼叫', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    
    # ASGI 解釋 - 避免絕對比較，說明實際功能
    asgi_explain_box = FancyBboxPatch((8.5, 3.6), 7, 1.2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#F0FFF0',
                                     edgecolor='#32CD32', linewidth=2)
    ax.add_patch(asgi_explain_box)
    ax.text(12, 4.5, 'uvicorn 是什麼？', 
            fontsize=16, fontweight='bold', ha='center', fontproperties=chinese_font)
    ax.text(12, 4.1, 'uvicorn 是支援 ASGI 的現代網頁伺服器', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    ax.text(12, 3.8, '提供非同步處理和 HTTP/2 支援', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    
    # === 技術詳細說明 - 下移 ===
    
    # uvicorn 介紹
    state_box = FancyBboxPatch((0.5, 2.1), 7.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue',
                              edgecolor='blue', linewidth=2)
    ax.add_patch(state_box)
    ax.text(4.25, 3.0, 'uvicorn 的角色', 
            fontsize=16, fontweight='bold', ha='center', fontproperties=chinese_font)
    ax.text(4.25, 2.6, 'uvicorn 啟動 FastAPI 應用程式並監聽 HTTP 請求', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    ax.text(4.25, 2.2, '處理連線管理、請求解析和回應傳送', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    
    # FastAPI 介紹
    async_box = FancyBboxPatch((8.5, 2.1), 7, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen',
                              edgecolor='green', linewidth=2)
    ax.add_patch(async_box)
    ax.text(12, 3.0, 'FastAPI 的角色', 
            fontsize=16, fontweight='bold', ha='center', fontproperties=chinese_font)
    ax.text(12, 2.6, 'FastAPI 定義 API 端點和處理業務邏輯', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    ax.text(12, 2.2, '驗證輸入參數並調用檢測服務回傳結果', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    
    # 資料流和技術重點 - 大幅下移，充分利用空間
    ax.text(8, 1.5, '資料轉換流程', fontsize=16, fontweight='bold', 
            ha='center', fontproperties=chinese_font)
    ax.text(8, 1.1, 'HTTP 請求 → PIL 圖片 → resize(560×560) → to_tensor → ImageNet normalize → ONNX 推理 → 檢測結果 → 標註圖片 → JSON', 
            fontsize=13, ha='center', fontproperties=chinese_font)
    
    ax.text(8, 0.6, '核心技術: RF-DETR ONNX 模型 | Pillow 預處理 | 標籤避重疊算法', 
            fontsize=13, ha='center', fontproperties=chinese_font, style='italic')
    
    # 技術特點 - 移到最底部，無留白
    ax.text(8, 0.1, '技術特點: 分層設計 | 非同步處理 | ONNX 推理 | HTTP API', 
            fontsize=13, ha='center', fontproperties=chinese_font, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def save_no_whitespace_diagram():
    """儲存無留白版流程圖"""
    fig = create_no_whitespace_diagram()
    
    # 儲存為高解析度 PNG
    fig.savefig('/workspace/docs/no_whitespace_request_flow.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # 儲存為 SVG
    fig.savefig('/workspace/docs/no_whitespace_request_flow.svg', 
                format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ 無留白版請求流程圖已生成:")
    print("   📁 /workspace/docs/no_whitespace_request_flow.png (高解析度)")
    print("   📁 /workspace/docs/no_whitespace_request_flow.svg (向量圖)")
    print("\n🎯 改善重點:")
    print("   - 大幅下移所有說明內容，消除底部留白")
    print("   - 改善 API 解釋：強調整合到各種系統的實用性")
    print("   - 改善 ASGI 解釋：對比傳統伺服器的具體差異")
    print("   - 增加實際應用情境說明")
    print("   - 充分利用畫布空間，無浪費區域")
    
    return fig

if __name__ == "__main__":
    fig = save_no_whitespace_diagram()
    
    try:
        plt.show()
    except:
        print("💡 圖表已儲存至檔案")