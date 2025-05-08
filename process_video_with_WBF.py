import warnings

from ultralytics import YOLO
import cv2
from tqdm import tqdm
from detectron2.model_zoo import model_zoo
from ensemble_boxes import weighted_boxes_fusion
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np


def count_vehicles_video(input_video_path, output_video_path):
    # 加载YOLO模型
    yolo_model = YOLO("YOLO_VisDrone.pt")

    # 加载Faster R-CNN模型
    faster_cfg = get_cfg()
    faster_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    faster_cfg.MODEL.WEIGHTS = "./model_final_280758.pkl"
    faster_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    faster_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    faster_model = DefaultPredictor(faster_cfg)

    # 视频输入输出设置
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 初始化跟踪相关变量
    tracked_vehicles = {}
    total_in = 0
    total_out = 0
    baseline_y = frame_height // 2

    # 跟踪器配置
    tracker_config = {
        "persist": True,
        "tracker": "bytetrack.yaml",
        "conf": 0.6,
        "iou": 0.5,
        "classes": None,
        "verbose": False
    }

    # 定义车辆类别映射
    vehicle_classes = [
        "bicycle", "car", "van", "truck",
        "tricycle", "awning-tricycle", "bus", "motor"
    ]

    # 禁用警告信息
    warnings.filterwarnings("ignore")

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 绘制基准线
            cv2.line(frame, (0, baseline_y), (frame_width, baseline_y), (0, 0, 255), 2)

            # YOLO推理
            yolo_results = yolo_model.predict(frame, conf=0.5, verbose=False)
            yolo_boxes = []
            yolo_scores = []
            yolo_labels = []
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls = int(box.cls.item())
                yolo_boxes.append([x1 / frame_width, y1 / frame_height, x2 / frame_width, y2 / frame_height])
                yolo_scores.append(conf)
                yolo_labels.append(cls)

            # Faster R-CNN推理
            faster_output = faster_model(frame)
            faster_boxes = []
            faster_scores = []
            faster_labels = []
            for box, score, cls in zip(faster_output["instances"].pred_boxes,
                                       faster_output["instances"].scores,
                                       faster_output["instances"].pred_classes):
                x1, y1, x2, y2 = box.cpu().numpy()
                faster_boxes.append([x1 / frame_width, y1 / frame_height, x2 / frame_width, y2 / frame_height])
                faster_scores.append(score.item())
                faster_labels.append(cls.item())

            # WBF融合
            boxes_list = [yolo_boxes, faster_boxes]
            scores_list = [yolo_scores, faster_scores]
            labels_list = [yolo_labels, faster_labels]
            weights = [2, 1]

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=0.5,
                skip_box_thr=0.4
            )

            # 转换回绝对坐标
            fused_boxes[:, 0] *= frame_width
            fused_boxes[:, 1] *= frame_height
            fused_boxes[:, 2] *= frame_width
            fused_boxes[:, 3] *= frame_height

            # 转换为检测结果格式用于跟踪
            detections = []
            for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, score, label])

            # 执行跟踪
            if len(detections) > 0:
                detections = np.array(detections)
                tracker_config["classes"] = [int(label) for label in detections[:, 5]]
                results = yolo_model.track(frame, **tracker_config)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                    # 更新跟踪状态
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if track_id not in tracked_vehicles:
                            tracked_vehicles[track_id] = {
                                "state": "above" if center_y < baseline_y else "below",
                                "history": []
                            }

                        tracked_vehicles[track_id]["history"].append(center_y)
                        if len(tracked_vehicles[track_id]["history"]) > 10:
                            tracked_vehicles[track_id]["history"].pop(0)

                        # 判断进出场
                        if tracked_vehicles[track_id]["state"] == "above" and all(
                                y >= baseline_y for y in tracked_vehicles[track_id]["history"][-3:]):
                            total_in += 1
                            tracked_vehicles[track_id]["state"] = "counted"
                        elif tracked_vehicles[track_id]["state"] == "below" and all(
                                y <= baseline_y for y in tracked_vehicles[track_id]["history"][-3:]):
                            total_out += 1
                            tracked_vehicles[track_id]["state"] = "counted"

                        # 绘制框和ID
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 显示统计信息
            info_text = [
                f"In: {total_in}",
                f"Out: {total_out}"
            ]
            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成 | 入场车辆数: {total_in}, 出场车辆数: {total_out}")
    print(f"结果保存至: {output_video_path}")


if __name__ == "__main__":
    count_vehicles_video("../video2.avi", "../video2_result_WBF.avi")
