import os
import cv2

# 定义需要保留的车辆类别ID（根据VisDrone官方类别定义）
# 定义类别映射
CATEGORY_MAP = {
    3: 0,  # Bicycle -> 0
    4: 1,  # Car -> 1
    5: 2,  # Van -> 2
    6: 3,  # Truck -> 3
    7: 4,  # Tricycle -> 4
    8: 5,  # Awning-tricycle -> 5
    9: 6,  # Bus -> 6
    10: 7  # Motor -> 7
}


def visdrone_to_yolo(visdrone_anno_path, yolo_anno_dir, images_dir):
    # 确保输出目录存在
    os.makedirs(yolo_anno_dir, exist_ok=True)

    # 遍历所有标注文件
    for anno_file in os.listdir(visdrone_anno_path):
        if not anno_file.endswith(".txt"):
            continue

        # 获取对应的图像尺寸
        image_name = anno_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        # 读取原始标注
        yolo_lines = []
        with open(os.path.join(visdrone_anno_path, anno_file), "r") as f:
            for line in f.readlines():
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                # 解析字段
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                category = int(parts[5])

                # 过滤非车辆目标
                if category not in CATEGORY_MAP:
                    continue

                # 计算归一化坐标并裁剪到 [0, 1]
                x_center: float = max(0.0, min(1.0, (bbox_left + bbox_w / 2) / img_width))
                y_center: float = max(0.0, min(1.0, (bbox_top + bbox_h / 2) / img_height))
                w: float = max(0.0, min(1.0, bbox_w / img_width))
                h: float = max(0.0, min(1.0, bbox_h / img_height))

                # 映射类别
                yolo_class = CATEGORY_MAP[category]
                yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # 写入YOLO格式标注
        output_path = os.path.join(yolo_anno_dir, anno_file)
        with open(output_path, "w") as f:
            f.writelines(yolo_lines)


# 调用示例（按需修改路径）
visdrone_to_yolo(
    visdrone_anno_path="../../VisDrone2019-DET-test-dev/annotations",
    yolo_anno_dir="../../VisDrone2019-YOLO/labels/test",
    images_dir="../../VisDrone2019-DET-test-dev/images"
)
