from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset

train_transform = transforms.Compose([
    # transforms.ToPILImage(), - Commented after first run as redownload is not required
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_transform = transforms.Compose([
    # transforms.ToPILImage(), - Commented after first run as redownload is not required
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


class AugmentationsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        return img1, img2, target

    def __len__(self):
        return len(self.dataset)