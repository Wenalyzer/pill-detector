def boxes_overlap(box1, box2):
    x1,y1,x2,y2 = box1
    ox1,oy1,ox2,oy2 = box2
    return not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2)

# 修復後的結果
detections = [
    ("Amoxicillin", [248,303,325,449], [242,449,349,492]),
    ("Diovan 160mg", [234,180,327,235], [228,137,368,180]),
    ("Takepron", [165,218,245,298], [68,212,165,255]),
    ("Relecox", [254,264,362,343], [248,216,330,259])
]

print("=== 修復後重疊檢查 ===")
found_overlap = False
for i in range(len(detections)):
    for j in range(i+1, len(detections)):
        if boxes_overlap(detections[i][2], detections[j][2]):
            print(f"❌ 標籤 {i+1} ({detections[i][0]}) 與標籤 {j+1} ({detections[j][0]}) 重疊")
            found_overlap = True

if not found_overlap:
    print("✅ 沒有標籤重疊")

print()
print("=== 標籤遮擋檢測框檢查 ===")
found_block = False
for i in range(len(detections)):
    for j in range(len(detections)):
        if i != j and boxes_overlap(detections[i][2], detections[j][1]):
            print(f"❌ 標籤 {i+1} ({detections[i][0]}) 遮擋檢測框 {j+1} ({detections[j][0]})")
            found_block = True

if not found_block:
    print("✅ 沒有標籤遮擋其他檢測框")

print()
print("=== 標籤位置分析 ===")
for i, (name, bbox, label_bbox) in enumerate(detections):
    x1, y1, x2, y2 = bbox
    lx1, ly1, lx2, ly2 = label_bbox
    
    # 判斷標籤位置
    if ly2 <= y1:
        position = "上方"
    elif ly1 >= y2:
        position = "下方"
    elif lx2 <= x1:
        position = "左側"
    elif lx1 >= x2:
        position = "右側"
    else:
        position = "重疊"
    
    print(f"{i+1}. {name}: {position}")