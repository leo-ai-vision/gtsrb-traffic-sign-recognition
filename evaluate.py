from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import (
    GTSRBDataset,
    collect_gtsrb_test_samples
)
import torch
import torch.nn as nn
from src.model import build_resnet18
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from PIL import Image

BATCH_SIZE = 64
IMAGE_SIZE = 128
NUM_CLASSES = 43

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"
# 添加结果图片路径
CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "confusion_matrix.png"
)
# 添加保存路径
MISCLASSIFIED_PATH = (
    RESULTS_DIR
    / "misclassified_examples.png"
)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_resnet18.pth"
)
# TEST_IMAGE_DIR → 图片文件夹
TEST_IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "GTSRB_Final_Test_Images"
    / "GTSRB"
    / "Final_Test"
    / "Images"
)
# TEST_CSV_PATH  → 标签CSV文件
TEST_CSV_PATH = (
    PROJECT_DIR
    / "data"
    / "GTSRB_Final_Test_GT"
    / "GT-final_test.csv"
)


def create_test_loader():
    # 读取测试图片路径和标签
    test_image_paths, test_labels = (
        collect_gtsrb_test_samples(
            test_image_dir=TEST_IMAGE_DIR,
            test_csv_path=TEST_CSV_PATH
        )
    )

    # 测试集不能使用随机数据增强
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = GTSRBDataset(
        image_paths=test_image_paths,
        labels=test_labels,
        transform=test_transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return test_loader

# 在整个测试集上计算Loss和准确率
def evaluate_model(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # 保存每张图片的预测结果和正确标签
    all_predictions = []
    all_labels = []

    with torch.no_grad():#关闭梯度函数
        for batch_index, (
            batch_images,
            batch_labels
        ) in enumerate(test_loader, start=1):

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

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                batch_labels.cpu().tolist()
            )

            if (
                batch_index % 50 == 0
                or batch_index == len(test_loader)
            ):
                print(
                    f"测试进度："
                    f"{batch_index}/{len(test_loader)}"
                )

    test_loss = total_loss / total_samples
    test_accuracy = total_correct / total_samples

    return (
        test_loss,
        test_accuracy,
        all_predictions,
        all_labels
    )

# 添加保存混淆矩阵的函数
# 生成并保存归一化混淆矩阵
def save_confusion_matrix(
    all_labels,
    all_predictions,
    save_path
):
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure, axis = plt.subplots(
        figsize=(16, 14)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true=all_labels,
        y_pred=all_predictions,
        labels=list(range(NUM_CLASSES)),
        normalize="true",
        cmap="Blues",
        include_values=False,
        ax=axis,
        colorbar=True
    )

    axis.set_title(
        "GTSRB Normalized Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)

# 找出错误分类图片并保存其中前12张
def save_misclassified_examples(
    image_paths,
    all_labels,
    all_predictions,
    save_path
):
    misclassified_indices = []

    # 找出真实类别与预测类别不同的图片位置
    for index, (
        true_label,
        predicted_label
    ) in enumerate(
        zip(all_labels, all_predictions)
    ):
        if true_label != predicted_label:
            misclassified_indices.append(index)

    selected_indices = misclassified_indices[:12]

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9)
    )

    # 把12张错误图片画到3行4列的画布中
    for axis, index in zip(
        axes.flat,
        selected_indices
    ):
        image = Image.open(
            image_paths[index]
        ).convert("RGB")

        true_label = all_labels[index]
        predicted_label = all_predictions[index]

        axis.imshow(image)

        axis.set_title(
            f"True: {true_label}\n"
            f"Predicted: {predicted_label}"
        )

        axis.axis("off")

    figure.suptitle(
        "GTSRB Misclassified Examples"
    )

    # 自动调整子图之间的间距，防止标题和图片重叠
    figure.tight_layout()
    # 创建保存图片所需的文件夹
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    # 将整张错误分类案例图保存到文件
    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )
    # 关闭图像，释放内存
    plt.close(figure)
    # 返回本次展示的错误分类样本数量
    return len(misclassified_indices)


def main():
    # 准备测试数据和计算设备
    test_loader = create_test_loader()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{CHECKPOINT_PATH}"
        )

    # 加载验证准确率最高的模型
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = build_resnet18(
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    print("计算设备：", device)
    print("测试图片数量：", len(test_loader.dataset))
    print("加载模型轮次：", checkpoint["epoch"])

    # 在全部官方测试图片上进行评价
    (
        test_loss,
        test_accuracy,
        all_predictions,
        all_labels
    ) = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # 计算43个类别的宏平均指标
    # Precision 表示：模型预测为某一类的图片中，有多少是真的。
    # Recall表示：某个类别所有真实图片中，模型成功找出了多少
    # F1是一个同时衡量Precision（精确率）和Recall（召回率）的指标。
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    # 保存测试集混淆矩阵
    save_confusion_matrix(
        all_labels=all_labels,
        all_predictions=all_predictions,
        save_path=CONFUSION_MATRIX_PATH
    )
    # 保存错误分类图片
    misclassified_count = (
        save_misclassified_examples(
            image_paths=(
                test_loader.dataset.image_paths
            ),
            all_labels=all_labels,
            all_predictions=all_predictions,
            save_path=MISCLASSIFIED_PATH
        )
    )

    print("测试完成")
    print(f"测试Loss：{test_loss:.4f}")
    print(f"测试Accuracy：{test_accuracy:.4f}")
    print(f"测试准确率：{test_accuracy * 100:.2f}%")
    print(f"宏平均Precision：{precision:.4f}")
    print(f"宏平均Recall：{recall:.4f}")
    print(f"宏平均F1：{f1:.4f}")
    print(
        "混淆矩阵已保存：",
        CONFUSION_MATRIX_PATH
    )
    print(
        "错误分类图片数量：",
        misclassified_count
    )

    print(
        "错误样本图已保存：",
        MISCLASSIFIED_PATH
    )


if __name__ == "__main__":
    main()