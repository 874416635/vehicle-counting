from ultralytics import YOLO
import cv2
from tqdm import tqdm

def count_vehicles_video(input_video_path, output_video_path):
    # 加载模型
    model = YOLO("YOLO_VisDrone.pt")

    # 获取类别名称映射
    class_names = model.names

    # 定义车辆类别（与图片处理保持一致）
    vehicle_classes = {
        class_id: class_name for class_id, class_name in class_names.items()
        if class_name.lower() in [
            "bicycle", "car", "van", "truck", "tricycle",
            "awning-tricycle", "bus", "motor"
        ]
    }

    # 视频输入输出设置
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 基准线的 y 坐标（视频中间位置）
    baseline_y = frame_height // 2

    # 初始化跟踪相关变量
    tracked_vehicles = {}  # 存储已追踪车辆ID及其状态 {"track_id": "state"}
    total_in = 0  # 累计入场计数器
    total_out = 0  # 累计出场计数器

    # 配置跟踪器参数
    tracker_config = {
        "persist": True,
        "tracker": "bytetrack.yaml",
        "conf": 0.6,
        "iou": 0.5,
        "classes": list(vehicle_classes.keys()),
        "verbose": False
    }

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 绘制基准线
            cv2.line(frame, (0, baseline_y), (frame_width, baseline_y), (0, 0, 255), 2)

            # 执行带追踪的预测
            results = model.track(frame, **tracker_config)

            if results[0].boxes.id is not None:
                # 获取检测信息
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # 处理每个检测目标
                for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                    if class_id in vehicle_classes:
                        # 获取检测框中心点坐标
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # 更新轨迹历史
                        if track_id not in tracked_vehicles:
                            tracked_vehicles[track_id] = {"state": "above" if center_y < baseline_y else "below",
                                                          "history": []}

                        tracked_vehicles[track_id]["history"].append(center_y)
                        if len(tracked_vehicles[track_id]["history"]) > 10:  # 保留最近 10 帧的轨迹
                            tracked_vehicles[track_id]["history"].pop(0)

                        # 判断轨迹方向
                        if tracked_vehicles[track_id]["state"] == "above" and all(
                                y >= baseline_y for y in tracked_vehicles[track_id]["history"][-3:]):
                            total_in += 1
                            tracked_vehicles[track_id]["state"] = "counted"
                        elif tracked_vehicles[track_id]["state"] == "below" and all(
                                y <= baseline_y for y in tracked_vehicles[track_id]["history"][-3:]):
                            total_out += 1
                            tracked_vehicles[track_id]["state"] = "counted"

                        # 绘制检测框和ID
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

            # 写入输出帧
            out.write(frame)

            # 更新进度条
            pbar.update(1)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成 | 入场车辆数: {total_in}, 出场车辆数: {total_out}")
    print(f"结果视频保存至: {output_video_path}")


if __name__ == "__main__":
    count_vehicles_video("../video1.mp4", "../video1_result_YOLO.mp4")
