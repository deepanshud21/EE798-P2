import torch
from torchvision import datasets, transforms

def get_dataloader(batch_size, train=True):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root='data/train' if train else 'data/test', transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
