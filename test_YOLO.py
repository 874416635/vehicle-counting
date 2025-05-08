from ultralytics import YOLO
from multiprocessing import freeze_support  # 导入freeze_support

if __name__ == '__main__':
    freeze_support()  # 确保Windows多进程兼容性
    model = YOLO("YOLO_VisDrone.pt")
    results = model.val(data="data_VisDrone_YOLO.yaml", split="test")