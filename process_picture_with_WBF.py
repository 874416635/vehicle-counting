import warnings

import cv2
from detectron2.model_zoo import model_zoo
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def count_vehicles_picture(input_path, output_path):
    # 加载YOLO模型
    yolo_model = YOLO("YOLO_VisDrone.pt")

    # 加载Faster R-CNN模型
    faster_cfg = get_cfg()
    faster_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    faster_cfg.MODEL.WEIGHTS = "./model_final_280758.pkl"
    faster_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    faster_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    faster_model = DefaultPredictor(faster_cfg)

    # 读取图像
    img = cv2.imread(input_path)
    height, width = img.shape[:2]

    # 禁用警告信息
    warnings.filterwarnings("ignore")

    # YOLO推理
    yolo_results = yolo_model.predict(img, conf=0.5)
    yolo_boxes = []
    yolo_scores = []
    yolo_labels = []
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf.item()
        cls = int(box.cls.item())
        yolo_boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
        yolo_scores.append(conf)
        yolo_labels.append(cls)

    # Faster R-CNN推理
    faster_output = faster_model(img)
    faster_boxes = []
    faster_scores = []
    faster_labels = []
    for box, score, cls in zip(faster_output["instances"].pred_boxes,
                               faster_output["instances"].scores,
                               faster_output["instances"].pred_classes):
        x1, y1, x2, y2 = box.cpu().numpy()
        faster_boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
        faster_scores.append(score.item())
        faster_labels.append(cls.item())

    # WBF融合
    boxes_list = [yolo_boxes, faster_boxes]
    scores_list = [yolo_scores, faster_scores]
    labels_list = [yolo_labels, faster_labels]
    weights = [2, 1]  # 给YOLO更高的权重

    fused_boxes, _, _ = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=0.5,
        skip_box_thr=0.4
    )

    # 转换回绝对坐标
    fused_boxes[:, 0] *= width
    fused_boxes[:, 1] *= height
    fused_boxes[:, 2] *= width
    fused_boxes[:, 3] *= height

    # 可视化和保存
    vehicle_classes = [
        "bicycle", "car", "van", "truck",
        "tricycle", "awning-tricycle", "bus", "motor"
    ]
    total = 0
    for box in fused_boxes:
        x1, y1, x2, y2 = map(int, box)
        total += 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(img, f"Total Vehicles: {total}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(output_path, img)
    print(f"结果保存至 {output_path}")
    print("总车辆数量:", total)


if __name__ == "__main__":
    count_vehicles_picture("../picture1.jpg", "../picture1_result_WBF.jpg")