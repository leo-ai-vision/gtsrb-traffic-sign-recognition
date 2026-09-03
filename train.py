import torch
import torch.nn as nn
from torch.optim import AdamW
from src.model import build_resnet18
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import (
    GTSRBDataset,
    collect_gtsrb_samples,
    split_gtsrb_samples
)
import matplotlib.pyplot as plt

NUM_CLASSES = 43
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent

# GTSRB训练集目录
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "GTSRB_Final_Training_Images"
    / "GTSRB"
    / "Final_Training"
    / "Images"
)


BATCH_SIZE = 64
IMAGE_SIZE = 128
VAL_RATIO = 0.2
RANDOM_SEED = 42
EPOCHS = 10
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_resnet18.pth"
# 增加曲线保存路径
RESULTS_DIR = PROJECT_DIR / "results"

TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"

# 创建训练和验证DataLoader
def create_data_loaders():
    image_paths, labels, group_ids = collect_gtsrb_samples(
        DATA_DIR
    )

    (
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels
    ) = split_gtsrb_samples(
        image_paths=image_paths,
        labels=labels,
        group_ids=group_ids,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = GTSRBDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        transform=train_transform
    )

    val_dataset = GTSRBDataset(
        image_paths=val_image_paths,
        labels=val_labels,
        transform=val_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader

# 完成一轮训练
def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (batch_images, batch_labels) in enumerate(
            train_loader
    ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        batch_size = batch_labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = outputs.argmax(dim=1)
        total_correct += (
            predictions == batch_labels
        ).sum().item()

        if (batch_index + 1) % 50 == 0:
            current_loss = total_loss / total_samples
            current_accuracy = total_correct / total_samples

            print(
                f"训练进度：{batch_index + 1}/{len(train_loader)}，"
                f"Loss：{current_loss:.4f}，"
                f"Accuracy：{current_accuracy:.4f}"
            )

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy

# 完成一轮验证
def validate_one_epoch(
        model,
        val_loader,
        criterion,
        device
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            batch_size = batch_labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)
            total_correct += (
                predictions == batch_labels
            ).sum().item()

    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples

    return val_loss, val_accuracy
# 绘制并保存训练过程曲线
def save_training_curves(
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        save_path
):
    epochs = range(1, len(train_losses) + 1)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # 绘制Loss曲线
    axes[0].plot(
        epochs,
        train_losses,
        marker="o",
        label="Train Loss"
    )
    axes[0].plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss"
    )
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 绘制Accuracy曲线
    axes[1].plot(
        epochs,
        train_accuracies,
        marker="o",
        label="Train Accuracy"
    )
    axes[1].plot(
        epochs,
        val_accuracies,
        marker="o",
        label="Validation Accuracy"
    )
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

def main():
    # 准备训练设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 创建模型
    model = build_resnet18(
        num_classes=NUM_CLASSES
    )
    model = model.to(device)

    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("计算设备：", device)
    print("模型输出类别：", model.fc.out_features)
    print("可训练参数数量：", trainable_parameters)
    print("损失函数：", criterion)
    print("优化器：", type(optimizer).__name__)

    # 准备训练和验证数据
    train_loader, val_loader = create_data_loaders()

    print("训练样本数量：", len(train_loader.dataset))
    print("验证样本数量：", len(val_loader.dataset))
    print("训练批次数量：", len(train_loader))
    print("验证批次数量：", len(val_loader))

    # 创建保存模型的文件夹
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # 读取已有最佳准确率，防止重新训练时覆盖更好的模型
    if BEST_MODEL_PATH.exists():
        saved_checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        best_val_accuracy = saved_checkpoint["val_accuracy"]

        print(
            f"当前模型最佳验证准确率："
            f"{best_val_accuracy:.4f}"
        )
    else:
        best_val_accuracy = 0.0
    # 记录每一轮的训练和验证指标
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 进行多轮训练和验证
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(f"训练Loss：{train_loss:.4f}")
        print(f"训练Accuracy：{train_accuracy:.4f}")
        print(f"验证Loss：{val_loss:.4f}")
        print(f"验证Accuracy：{val_accuracy:.4f}")

        # 保存当前轮次的指标，后面用于绘制曲线
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 当前验证准确率更高时，覆盖保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),#保存模型已经学习到的参数，也就是 ResNet18 各层的权重和偏置。
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                BEST_MODEL_PATH
            )

            print(f"已保存最佳模型：{BEST_MODEL_PATH}")

    # 训练全部结束后保存曲线
    save_training_curves(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        save_path=TRAINING_CURVES_PATH
    )

    print("训练曲线已保存：", TRAINING_CURVES_PATH)

    print(f"\n训练结束，最佳验证准确率：{best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()