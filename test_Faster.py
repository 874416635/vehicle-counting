from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# 注册VisDrone测试集
def register_visdrone_testset():
    thing_classes = [
        "bicycle", "car", "van", "truck",
        "tricycle", "awning-tricycle", "bus", "motor"
    ]
    register_coco_instances(
        "visdrone_test",
        {"thing_classes": thing_classes},
        "../VisDrone2019-Faster/annotations/test.json",  # 测试集标注路径
        "../VisDrone2019-Faster/images/test"  # 测试集图片路径
    )


if __name__ == "__main__":
    # 注册测试集
    register_visdrone_testset()

    # 获取元数据
    metadata = MetadataCatalog.get("visdrone_test")
    print(f"测试集类别: {metadata.thing_classes}")

    # 配置模型
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "./Faster_VisDrone.pth"  # 训练好的模型路径
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试阈值
    cfg.DATASETS.TEST = ("visdrone_test",)

    # 初始化预测器
    predictor = DefaultPredictor(cfg)

    # 构建测试数据加载器
    test_loader = build_detection_test_loader(cfg, "visdrone_test")

    # 初始化评估器
    evaluator = COCOEvaluator(
        "visdrone_test",
        output_dir="./output/visdrone_faster_rcnn/test_results"  # 测试结果保存路径
    )

    # 执行评估
    metrics = inference_on_dataset(predictor.model, test_loader, evaluator)

    # 打印关键指标
    print("\n=== 测试结果 ===")
    print(f"mAP@[0.5:0.95]: {metrics['bbox']['AP']:.3f}")
    print(f"AP50: {metrics['bbox']['AP50']:.3f}")
    print(f"AP75: {metrics['bbox']['AP75']:.3f}")
