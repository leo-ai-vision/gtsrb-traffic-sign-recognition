import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# 创建用于GTSRB分类的ResNet18
def build_resnet18(num_classes=43):
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features=input_features,
        out_features=num_classes
    )

    return model