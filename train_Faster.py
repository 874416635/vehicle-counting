import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator


# 注册数据集
def register_visdrone_datasets():
    # 手动定义与COCO标注文件一致的类别列表
    thing_classes = [
        "bicycle", "car", "van", "truck",
        "tricycle", "awning-tricycle", "bus", "motor"
    ]

    # 注册训练集（显式传递元数据）
    register_coco_instances(
        "visdrone_train",
        {"thing_classes": thing_classes},
        "../VisDrone2019-Faster/annotations/train.json",
        "../VisDrone2019-Faster/images/train"
    )

    # 验证集同理
    register_coco_instances(
        "visdrone_val",
        {"thing_classes": thing_classes},
        "../VisDrone2019-Faster/annotations/val.json",
        "../VisDrone2019-Faster/images/val"
    )


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # 显式传递参数替代cfg [[3]][[7]]
        return COCOEvaluator(
            dataset_name=dataset_name,  # 数据集名称
            tasks=("bbox",),  # 仅评估目标检测任务
            distributed=False,  # 非分布式训练
            output_dir=output_folder or cfg.OUTPUT_DIR
        )


if __name__ == "__main__":
    register_visdrone_datasets()

    # 获取类别元数据
    metadata = MetadataCatalog.get("visdrone_train")
    num_classes = len(metadata.thing_classes)  # 应为8类（根据你的target_classes）

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("visdrone_train",)
    cfg.DATASETS.TEST = ("visdrone_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "./model_final_280758.pkl"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"  # 余弦学习率衰减
    cfg.TEST.EVAL_PERIOD = 2000  # 每2000次迭代评估一次验证集
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000  # 每2000次保存一次模型
    cfg.SOLVER.AMP.ENABLED = False  # 混合精度训练
    cfg.OUTPUT_DIR = "./output/visdrone_faster_rcnn1"

    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]  # 调整锚点尺寸
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]  # 调整宽高比
    # cfg.INPUT.MIN_SIZE_TRAIN = (640, 672)
    # cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # cfg.INPUT.MIN_SIZE_TEST = 800
    # cfg.INPUT.MAX_SIZE_TEST = 1333

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
