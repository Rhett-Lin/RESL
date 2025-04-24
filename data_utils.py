import faiss
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder



BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(dataset="CIFAR-10"):
    # 这个函数用于获取指定数据集的转换操作
    # 可用于调整图像大小，带有中心剪裁，将图像转换为RGB格式，以及标准化等转换流程。

    if (
        dataset == "CIFAR-10"
        or dataset == "CIFAR-20"
        or dataset == "STL-10"
        or dataset == "DTD"
        or dataset == "UCF101"
        or dataset == "food-101"
        or dataset == "CIFAR-20-test"
    ):
        # 如果数据集是 CIFAR-10, CIFAR-20, STL-10, DTD, UCF101 之一
        transforms = torchvision.transforms.Compose(
            [
                # 调整图像大小为 224x224，使用 BICUBIC 插值
                torchvision.transforms.Resize(224, interpolation=BICUBIC),
                # 中心剪裁图像，大小为 224x224
                torchvision.transforms.CenterCrop(224),
                # 将图像转换为 RGB 格式
                _convert_image_to_rgb,
                # 将图像转换为 Tensor，值基于 [0, 1]
                torchvision.transforms.ToTensor(),
                # 对图像进行标准化操作，每个颜色速度通道都有平均和标准差
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),  # 平均值
                    (0.26862954, 0.26130258, 0.27577711),  # 标准差
                ),
            ]
        )       
    elif (
        dataset == "ImageNet-Dogs" or dataset == "ImageNet-10" or dataset == "ImageNet" or dataset == "tiny-imagenet-200" or dataset == "ImageNet-1K" or dataset == "Oxford-102"
    ):
        # 如果数据集是 ImageNet-Dogs, ImageNet-10, ImageNet, tiny-imagenet-200 之一
        transforms = torchvision.transforms.Compose(
            [
                # 调整图像大小为 256x256，使用 BICUBIC 插值
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                # 中心剪裁图像，大小为 224x224
                torchvision.transforms.CenterCrop(224),
                # 将图像转换为 RGB 格式
                _convert_image_to_rgb,
                # 将图像转换为 Tensor，值基于 [0, 1]
                torchvision.transforms.ToTensor(),
                # 对图像进行标准化操作，每个颜色速度通道都有平均和标准差
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),  # 平均值
                    (0.26862954, 0.26130258, 0.27577711),  # 标准差
                ),
            ]
        )
    else:
        # 如果输入的数据集名称不在支持范围内，抛出未实现的错误
        raise NotImplementedError
    return transforms  # 返回应用于该数据集的转换流程



def get_dataloader(dataset="CIFAR-10", batch_size=4096):
    transforms = get_transforms(dataset)
    if dataset == "CIFAR-10":
        data_train = CIFAR10(
            root="/home/zixuanlin/data/data", train=True, download=True, transform=transforms
        )
        data_test = CIFAR10(
            root="/home/zixuanlin/data/data", train=False, download=True, transform=transforms
        )
    elif dataset == "CIFAR-20-test":
        data_train = CIFAR100(
            root="/home/zixuanlin/data/data", train=True, download=True, transform=transforms
        )
        data_test = CIFAR100(
            root="/home/zixuanlin/data/data", train=False, download=True, transform=transforms
        )
    elif dataset == "CIFAR-20":
        data_train = CIFAR100(
            root="/home/zixuanlin/data/data", train=True, download=True, transform=transforms
        )
        data_test = CIFAR100(
            root="/home/zixuanlin/data/data", train=False, download=True, transform=transforms
        )
    elif dataset == "STL-10":
        data_train = STL10(
            root="/home/zixuanlin/data/data", split="train", download=True, transform=transforms
        )
        data_test = STL10(
            root="/home/zixuanlin/data/data", split="test", download=True, transform=transforms
        )
    elif dataset == "Oxford-102":
        data_train = ImageFolder("/home/zixuanlin/data/data/Oxford 102 Flower/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/Oxford 102 Flower/test_val", transform=transforms)
    elif dataset == "ImageNet-10":
        data_train = ImageFolder("/home/zixuanlin/data/data/ImageNet-10/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/ImageNet-10/test", transform=transforms)
        # subset_file = "/home/zixuanlin/data/data/imagenet_subsets/imagenet_10.txt"
        # data_train = ImageNetSubset(subset_file=subset_file, split='train', transform=transforms)
        # data_test = ImageNetSubset(subset_file=subset_file, split='val', transform=transforms)
    elif dataset == "ImageNet-Dogs":
        data_train = ImageFolder("/home/zixuanlin/data/data/imagenet-dog/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/imagenet-dog/val", transform=transforms)
    elif dataset == "DTD":
        data_train = ImageFolder("/home/zixuanlin/data/data/DTD/dtd/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/DTD/dtd/test", transform=transforms)
    elif dataset == "UCF101":
        data_train = ImageFolder("/home/zixuanlin/data/data/UCF101/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/UCF101/val", transform=transforms)
    elif dataset == "ImageNet":      
        data_train = ImageFolder("/home/zixuanlin/data/data/ImageNet/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/ImageNet/val", transform=transforms)
    elif dataset == "tiny-imagenet-200":
        data_train = ImageFolder("/home/zixuanlin/data/data/tiny-imagenet-200/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/tiny-imagenet-200/val", transform=transforms)
    elif dataset == "food-101":
        data_train = ImageFolder("/home/zixuanlin/data/data/food-101/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/data/food-101/test", transform=transforms)
    elif dataset == "ImageNet-1K":
        data_train = ImageFolder("/home/zixuanlin/data/imagenet-1k/train", transform=transforms)
        data_test = ImageFolder("/home/zixuanlin/data/imagenet-1k/val", transform=transforms)
    else:
        raise NotImplementedError

    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dataloader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return dataloader_train, dataloader_test


def mine_nearest_neighbors(features, topk=50):
    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    print("Nearest neighbors computed.")
    return indices[:, 1:]


class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image):
        super(NeighborsDataset, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, neighbor_text, neighbor_image
