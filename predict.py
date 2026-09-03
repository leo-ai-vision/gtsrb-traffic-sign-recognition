import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
from src.model import build_resnet18
from src.class_names import GTSRB_CLASS_NAMES
import time


PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_IMAGE_PATH = (
    PROJECT_DIR
    / "data"
    / "GTSRB_Final_Test_Images"
    / "GTSRB"
    / "Final_Test"
    / "Images"
    / "00000.ppm"
)
IMAGE_SIZE = 128
NUM_CLASSES = 43

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_resnet18.pth"
)

# 读取命令行中的图片路径参数
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GTSRB交通标志单张图片预测"
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="需要预测的交通标志图片路径"
    )

    return parser.parse_args()

# 将单张图片转换成模型需要的Tensor
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = image_transform(image)

    # 添加批次维度：[3, 128, 128]变成[1, 3, 128, 128]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# 加载训练完成的最佳模型
def load_trained_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{CHECKPOINT_PATH}"
        )

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
    model.eval()

    return model, checkpoint["epoch"]

def main():
    args = parse_arguments()
    image_path = args.image

    # 检查用户提供的图片是否存在
    if not image_path.exists():
        raise FileNotFoundError(
            f"找不到图片：{image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"提供的路径不是文件：{image_path}"
        )
    # 读取并预处理单张图片
    image_tensor = preprocess_image(image_path)

    # 加载模型并把图片移动到相同设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, saved_epoch = load_trained_model(device)
    image_tensor = image_tensor.to(device)

    # 预测类别和置信度
    with torch.no_grad():
        # 先预热一次，避免首次CUDA初始化影响计时
        # 第一次 → 预热，不计时
        # 第二次 → 正式预测，计时
        model(image_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()#会让CPU等待，直到GPU完成之前的任务。
        # 开始计时
        start_time = time.perf_counter()
        # 第二次模型计算：正式测量
        outputs = model(image_tensor)
        # 等待正式计算完成
        if device.type == "cuda":
            torch.cuda.synchronize()
        # 结束计时
        inference_time_ms = (
                                    time.perf_counter() - start_time
                            ) * 1000

        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        # 找出最大概率及其类别
        confidence, predicted_class = (
            probabilities.max(dim=1)
        )
        # 取得概率最高的5个类别
        top_probabilities, top_classes = (
            probabilities.topk(
                k=5,
                dim=1
            )
        )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    top_probabilities = (
        top_probabilities[0].cpu().tolist()
    )

    top_classes = (
        top_classes[0].cpu().tolist()
    )

    predicted_name = GTSRB_CLASS_NAMES[
        predicted_class
    ]

    print("图片参数读取成功")
    print("待预测图片：", image_path.resolve())
    print("图片Tensor形状：", image_tensor.shape)
    print("图片数据类型：", image_tensor.dtype)
    print("计算设备：", device)
    print("加载模型轮次：", saved_epoch)
    print("模型输出形状：", outputs.shape)
    print("预测类别ID：", predicted_class)
    print(f"预测置信度：{confidence * 100:.2f}%")
    print("预测类别名称：", predicted_name)
    print(
        f"推理时间：{inference_time_ms:.2f} ms"
    )
    print("\nTop-5预测结果：")

    for rank, (
        class_id,
        probability
    ) in enumerate(
        zip(top_classes, top_probabilities),
        start=1
    ):
        class_name = GTSRB_CLASS_NAMES[class_id]

        print(
            f"{rank}. "
            f"{class_name} "
            f"(类别{class_id})："
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()