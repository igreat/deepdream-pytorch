# feature extraction

from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(
        self,
        layers=["relu4_1", "relu5_1"],
        device="cpu",
    ):
        super(VGG19, self).__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        features.requires_grad_(False)

        self.slices = []
        slice_temp = nn.Sequential()
        pool_cnt, relu_count, conv_count = 1, 1, 1
        for i in range(len(features)):
            x = features[i]
            if isinstance(x, nn.Conv2d):
                name = f"conv{pool_cnt}_{conv_count}"
                conv_count += 1
            elif isinstance(x, nn.ReLU):
                name = f"relu{pool_cnt}_{relu_count}"
                relu_count += 1
            else:
                name = f"pool{pool_cnt}"

                relu_count = 1
                conv_count = 1
                pool_cnt += 1

            slice_temp.add_module(name, x)
            if name in layers:
                self.slices.append(slice_temp)
                slice_temp = nn.Sequential()
                layers.remove(name)

            # making sure it is cut off at the last loss layer to avoid unnecesarry computations
            if len(layers) == 0:
                break

    def forward(self, input):
        activations = []
        x = input
        for slice in self.slices:
            x = slice(x)
            activations.append(x.mean())

        return activations
