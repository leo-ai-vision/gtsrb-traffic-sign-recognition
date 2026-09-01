from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset import (
    GTSRBDataset,
    collect_gtsrb_samples,
    split_gtsrb_samples
)


train_dir = Path(
    r"D:\python\traffic-sign-adversarial-robustness\data\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images"
)
# 扫描数据集并建立样本信息
image_paths, labels, group_ids = collect_gtsrb_samples(train_dir)

num_classes = len(set(labels))

print("类别数量：", num_classes)
print("图片数量：", len(image_paths))
print("标签数量：", len(labels))
print("拍摄序列数量：", len(set(group_ids)))
print("第一个样本：", image_paths[0])
print("第一个标签：", labels[0])
print("第一个样本序列：", group_ids[0])
print("最后一个样本：", image_paths[-1])
print("最后一个标签：", labels[-1])
print("最后一个样本序列：", group_ids[-1])

# 按拍摄序列划分训练集和验证集
(
    train_image_paths,
    train_labels,
    val_image_paths,
    val_labels
) = split_gtsrb_samples(
    image_paths=image_paths,
    labels=labels,
    group_ids=group_ids,
    val_ratio=0.2,
    seed=42
)

print("训练图片数量：", len(train_image_paths))
print("训练标签数量：", len(train_labels))
print("验证图片数量：", len(val_image_paths))
print("验证标签数量：", len(val_labels))
print("划分后的图片总数：", len(train_image_paths) + len(val_image_paths))
print("训练集类别数量：", len(set(train_labels)))
print("验证集类别数量：", len(set(val_labels)))

# 直接从已经建立的列表中获取第一个样本
first_image_path = image_paths[0]
first_label = labels[0]

image = Image.open(first_image_path).convert("RGB")

# 训练集：预处理、数据增强和归一化
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(
        degrees=10,## 随机旋转-10°到+10°
        translate=(0.05, 0.05),#允许图片在水平和垂直方向随机移动最多5%。
        scale=(0.9, 1.1)#随机缩放到原来的90%～110%
    ),
    transforms.ColorJitter(
        brightness=0.2,#亮度大约在原来的80%～120%之间变化，模拟阴天、晴天或光照变化。
        contrast=0.2,#对比度在80%～120%之间变化，模拟图片清晰程度和光线反差变化。
        saturation=0.2#颜色饱和度在80%～120%之间变化，模拟相机、天气和标志褪色造成的颜色差异。
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 验证集：只进行固定预处理和归一化
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 创建训练集和验证集
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

print("训练Dataset数量：", len(train_dataset))
print("验证Dataset数量：", len(val_dataset))

# Dataset：规定怎样读取一张图片和标签
# DataLoader：一次读取多张图片，并组成训练批次
# 创建训练和验证DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# 分别取出一个批次进行检查
train_batch_images, train_batch_labels = next(iter(train_loader))
val_batch_images, val_batch_labels = next(iter(val_loader))

print("训练批次图片形状：", train_batch_images.shape)
print("训练批次标签形状：", train_batch_labels.shape)
print("验证批次图片形状：", val_batch_images.shape)
print("验证批次标签形状：", val_batch_labels.shape)



plt.imshow(image)
plt.title(f"Class: {first_label}")
plt.axis("off")
plt.show()
