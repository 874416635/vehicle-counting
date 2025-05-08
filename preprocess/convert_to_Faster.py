import os
import json
import cv2
from tqdm import tqdm

# 定义完整的类别映射
FULL_CATEGORY_MAP = {
    3: {"id": 1, "name": "bicycle"},
    4: {"id": 2, "name": "car"},
    5: {"id": 3, "name": "van"},
    6: {"id": 4, "name": "truck"},
    7: {"id": 5, "name": "tricycle"},
    8: {"id": 6, "name": "awning-tricycle"},
    9: {"id": 7, "name": "bus"},
    10: {"id": 8, "name": "motor"}
}


def visdrone_to_coco(visdrone_root, output_json, target_classes):
    # 默认保留全部车辆类别（3/4/5/6/7/8/9/10）
    if target_classes is None:
        target_classes = list(FULL_CATEGORY_MAP.keys())

    # 初始化COCO数据结构（动态生成categories）
    coco_data = {
        "info": {"description": "VisDrone2019-DET in COCO format"},
        "licenses": [],
        "categories": [
            {"id": v["id"], "name": v["name"]}
            for k, v in FULL_CATEGORY_MAP.items()
            if k in target_classes
        ],
        "images": [],
        "annotations": []
    }

    # 遍历图像和标注
    image_id = 0
    annotation_id = 0
    image_dir = os.path.join(visdrone_root, "images")
    anno_dir = os.path.join(visdrone_root, "annotations")

    for img_file in tqdm(sorted(os.listdir(image_dir))):
        if not img_file.lower().endswith((".jpg", ".png")):
            continue

        # 读取图像尺寸
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的图像: {img_path}")
            continue
        height, width = img.shape[:2]

        # 添加图像信息
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # 处理对应标注文件
        anno_path = os.path.join(anno_dir, os.path.splitext(img_file)[0] + ".txt")
        if not os.path.exists(anno_path):
            continue

        with open(anno_path, "r") as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                try:
                    # 解析字段
                    x1 = max(0.0, float(parts[0]))
                    y1 = max(0.0, float(parts[1]))
                    w = max(0.0, float(parts[2]))
                    h = max(0.0, float(parts[3]))
                    cat_id = int(parts[5])

                    # 跳过无效坐标
                    if x1 >= width or y1 >= height:
                        continue

                    # 调整宽高不越界
                    w = min(w, width - x1)
                    h = min(h, height - y1)
                    if w <= 0 or h <= 0:
                        continue

                except (ValueError, IndexError) as e:
                    print(f"标注解析失败: {anno_path} 第{line_idx}行 - {str(e)}")
                    continue

                # 过滤类别
                if cat_id not in target_classes or cat_id not in FULL_CATEGORY_MAP:
                    continue

                # 生成多边形坐标
                seg_poly = [x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h]

                # 添加到标注
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": FULL_CATEGORY_MAP[cat_id]["id"],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [seg_poly]
                })
                annotation_id += 1

        image_id += 1

    # 保存为JSON文件
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)


# 调用示例（根据需求选择保留的类别）
visdrone_to_coco(
    visdrone_root="../../VisDrone2019-DET-train/VisDrone2019-DET-train",
    output_json="../../VisDrone2019-Faster/annotations/train.json",
    target_classes=[3, 4, 5, 6, 7, 8, 9, 10]  # 保留所有车辆类别
)

visdrone_to_coco(
    visdrone_root="../../VisDrone2019-DET-val/VisDrone2019-DET-val",
    output_json="../../VisDrone2019-Faster/annotations/val.json",
    target_classes=[3, 4, 5, 6, 7, 8, 9, 10]  # 保留所有车辆类别
)

visdrone_to_coco(
    visdrone_root="../../VisDrone2019-DET-test-dev",
    output_json="../../VisDrone2019-Faster/annotations/test.json",
    target_classes=[3, 4, 5, 6, 7, 8, 9, 10]  # 保留所有车辆类别
)
