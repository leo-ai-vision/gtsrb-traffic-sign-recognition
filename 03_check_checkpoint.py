from pathlib import Path

import torch

from src.model import build_resnet18


NUM_CLASSES = 43

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "best_resnet18.pth"
)


def main():
    # 检查模型文件是否存在
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 读取训练检查点
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    # 创建相同结构的ResNet18并加载训练参数
    model = build_resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("模型加载成功")
    print("模型路径：", CHECKPOINT_PATH)
    print("保存轮次：", checkpoint["epoch"])
    print(f"验证Loss：{checkpoint['val_loss']:.4f}")
    print(
        f"验证Accuracy："
        f"{checkpoint['val_accuracy']:.4f}"
    )
    print("计算设备：", device)


if __name__ == "__main__":
    main()