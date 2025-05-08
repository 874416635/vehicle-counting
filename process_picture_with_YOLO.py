from ultralytics import YOLO
import cv2


def count_vehicles_picture(input_path,  output_path):
    # 加载模型
    model = YOLO("YOLO_VisDrone.pt")

    # 推理
    results = model.predict(input_path, conf=0.5)

    # 获取类别名称映射（从模型直接读取）
    class_names = results[0].names

    # 定义需要统计的车辆类别（动态获取类别名称）
    vehicle_classes = {
        class_id: class_name for class_id, class_name in class_names.items()
        if class_name.lower() in [
            "bicycle", "car", "van", "truck", "tricycle",
            "awning-tricycle", "bus", "motor"
        ]
    }

    # 初始化计数器
    count_dict = {name: 0 for name in vehicle_classes.values()}
    total = 0

    # 统计检测结果
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for class_id in class_ids:
        if class_id in vehicle_classes:
            class_name = vehicle_classes[class_id]
            count_dict[class_name] += 1
            total += 1

    # 绘制检测框和计数
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法加载图片: {input_path}")
        return

    # 统一框的颜色为绿色
    default_color = (0, 255, 0)  # 绿色

    # 绘制每个检测框
    for box, class_id in zip(boxes, class_ids):
        if class_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            color = default_color  # 使用统一的绿色框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 在左上角显示统计信息
    y_offset = 30
    cv2.putText(img, f"Total Vehicles: {total}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"结果保存至 {output_path}")
    print("总车辆数量:", total)


if __name__ == "__main__":
    count_vehicles_picture("../picture1.jpg", "../picture1_result_YOLO.jpg")  # 输入图片路径
