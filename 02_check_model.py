import torch
import torch.nn as nn
from src.model import build_resnet18


# 选择GPU或CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 创建模型并移动到计算设备
model = build_resnet18(num_classes=43)
model = model.to(device)
model.eval()

# 创建4张模拟图片
dummy_images = torch.randn(
    4,
    3,
    128,
    128,
    device=device
)

# 测试模型前向传播
with torch.no_grad():
    outputs = model(dummy_images)

# 创建模拟标签并计算分类损失
dummy_labels = torch.tensor(
    [0, 1, 2, 3],
    device=device
)

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, dummy_labels)

print("模拟标签形状：", dummy_labels.shape)
print("分类损失：", loss.item())

print("计算设备：", device)
print("模型最后一层：", model.fc)
print("输入形状：", dummy_images.shape)
print("输出形状：", outputs.shape)