import os
import random
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# 加载 COCO 数据集
coco = COCO("../../VisDrone2019-Faster/annotations/train.json")

# 获取所有图像 ID
all_img_ids = coco.getImgIds()

# 随机挑选一张图片
img_id = random.choice(all_img_ids)
img_info = coco.loadImgs(img_id)[0]

# 构建图像路径并检查文件是否存在
image_path = os.path.normpath(os.path.join("../../VisDrone2019-DET-train/VisDrone2019-DET-train/images", img_info["file_name"]))
print("Image Path:", image_path)

if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
    exit(1)

# 加载图像并检查是否成功
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image. Check the file path or integrity.")
    exit(1)

# 加载标注信息
anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

# 显示图像并绘制边界框
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for ann in anns:
    x, y, w, h = map(int, ann["bbox"])  # 确保 bbox 坐标为整数
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red'))
plt.show()
