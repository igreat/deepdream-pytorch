import torch
from torchvision import transforms
from torch import nn

VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD = torch.tensor([0.229, 0.224, 0.225])


def normalize(image):
    return transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)(image)


def denormalize(image):
    return transforms.Normalize(mean=-VGG_MEAN, std=1 / VGG_STD)(image)


def preprocess(image, image_size=512):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
        ]
    )
    return transform(image).unsqueeze(0)


# Total variation loss
def tv_loss(image):
    x_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
