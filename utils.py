import torch
from torchvision import transforms
from torch import nn

VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


def preprocess(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(512),
            transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
        ]
    )
    return transform(image).unsqueeze(0)


# Total variation loss
class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.weight * (
            torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff))
        )
        return input
