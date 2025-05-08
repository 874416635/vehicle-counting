import torch
print(torch.__version__)           # 查看 PyTorch 版本
print(torch.cuda.is_available())   # 检查 GPU 是否可用

import torchvision
print(torchvision.__version__)  # 应输出 >=0.9.0

import torch
from torchvision.ops import nms

# 生成测试数据
boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9], dtype=torch.float32).cuda()

# 尝试在GPU上执行NMS
try:
    nms(boxes, scores, iou_threshold=0.5)
    print("NMS on GPU works!")
except:
    print("NMS on GPU failed!")