from ultralytics import YOLO


def main():
    # 加载预训练模型
    model = YOLO("yolo11m.pt")

    # 启动训练流程
    results = model.train(
        data="data_VisDrone_YOLO.yaml",
        epochs=100,
        imgsz=640,
        batch=8,  # 默认批次大小(16)
        device='cuda:0',  # 自动检测GPU可用性
        # resume=True  # 恢复训练
    )


if __name__ == "__main__":
    main()
