from pycocotools.coco import COCO

coco = COCO("../../VisDrone2019-Faster/annotations/train.json")
print(coco.dataset)  # 无报错则表示格式正确