from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import random
import csv
from src.model import build_resnet18

class GTSRBDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        if len(self.image_paths) != len(self.labels):
            raise ValueError("图片数量与标签数量不一致")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """读取并返回指定索引的图片和标签。"""

        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# 扫描GTSRB训练目录，建立图片、标签和拍摄序列列表
def collect_gtsrb_samples(train_dir):
    train_dir = Path(train_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"找不到训练集：{train_dir}")

    class_dirs = []

    for path in train_dir.iterdir():
        if path.is_dir():
            class_dirs.append(path)

    class_dirs = sorted(class_dirs)

    image_paths = []
    labels = []
    group_ids = []

    for class_dir in class_dirs:
        label = int(class_dir.name)
        class_images = sorted(class_dir.glob("*.ppm"))

        for image_path in class_images:
            track_id = image_path.stem.split("_")[0]

            image_paths.append(image_path)
            labels.append(label)
            group_ids.append((label, track_id))

    return image_paths, labels, group_ids

# 按拍摄序列划分训练集和验证集
def split_gtsrb_samples(
        image_paths,
        labels,
        group_ids,
        val_ratio=0.2,
        seed=42
):
    if not (len(image_paths) == len(labels) == len(group_ids)):
        raise ValueError("图片、标签和序列数量不一致")

    random_generator = random.Random(seed)

    all_groups = sorted(set(group_ids))
    unique_labels = sorted(set(labels))

    train_groups = set()
    val_groups = set()

    for label in unique_labels:
        class_groups = []

        for group_id in all_groups:
            if group_id[0] == label:
                class_groups.append(group_id)

        random_generator.shuffle(class_groups)

        val_count = max(
            1,
            round(len(class_groups) * val_ratio)
        )

        val_groups.update(class_groups[:val_count])
        train_groups.update(class_groups[val_count:])

    train_image_paths = []
    train_labels = []
    val_image_paths = []
    val_labels = []

    for image_path, label, group_id in zip(
            image_paths,
            labels,
            group_ids
    ):
        if group_id in train_groups:
            train_image_paths.append(image_path)
            train_labels.append(label)

        elif group_id in val_groups:
            val_image_paths.append(image_path)
            val_labels.append(label)

        else:
            raise ValueError(f"样本没有被划分：{image_path}")

    return (
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels
    )

# 读取GTSRB官方测试集图片路径和标签
def collect_gtsrb_test_samples(test_image_dir, test_csv_path):
    test_image_dir = Path(test_image_dir)
    test_csv_path = Path(test_csv_path)

    if not test_image_dir.exists():
        raise FileNotFoundError(
            f"找不到测试图片目录：{test_image_dir}"
        )

    if not test_csv_path.exists():
        raise FileNotFoundError(
            f"找不到测试标签文件：{test_csv_path}"
        )

    image_paths = []
    labels = []

    # CSV文件使用分号分隔
    with test_csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")

        for row in reader:
            image_path = test_image_dir / row["Filename"]
            label = int(row["ClassId"])

            image_paths.append(image_path)
            labels.append(label)

    return image_paths, labels