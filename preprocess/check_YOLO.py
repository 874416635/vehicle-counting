# 可视化随机样本（确保标注框与图像匹配）
import cv2
import random
import os


def plot_yolo_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        for line in f.readlines():
            class_id, xc, yc, bw, bh = map(float, line.strip().split())
            # 转换为像素坐标
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Check", img)
    cv2.waitKey(0)


# 随机检查一个样本
image_dir = "../../VisDrone2019-YOLO/images/test"
label_dir = "../../VisDrone2019-YOLO/labels/test"
all_images = os.listdir(image_dir)
random_image = random.choice(all_images)
plot_yolo_boxes(os.path.join(image_dir, random_image), os.path.join(label_dir, random_image.replace(".jpg", ".txt")))
