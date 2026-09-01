import torch
import torchvision

print("PyTorch版本：", torch.__version__)
print("torchvision版本：", torchvision.__version__)
print("CUDA是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡：", torch.cuda.get_device_name(0))