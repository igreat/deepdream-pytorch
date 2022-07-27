import torch
from torchvision.models import vgg19, VGG19_Weights
import utils
from PIL import Image
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image

device = "mps"

# training step
def train(gen_img, loss_model, optimizer):
    pass


def main():
    loss_model = vgg19(weights=VGG19_Weights.DEFAULT).eval().to(device)
    loss_model.requires_grad_(False)

    # image = Image.open("test-img.png")
    # image = utils.preprocess(image).to(device)
    # print(torch.argmax(F.softmax(loss_model(image))))

    tv_loss = utils.TVLoss(1)
    gen_img = torch.randn(size=(1, 3, 512, 512)).to(device)

    optimizer = optim.LBFGS(
        [gen_img], max_iter=200, lr=1e-1, tolerance_change=-1, tolerance_grad=-1
    )
    gen_img.requires_grad_(True)
    step = [0]

    def optim_step():
        optimizer.zero_grad()

        target_loss = -1e2 * loss_model(gen_img).squeeze(0)[207]
        tv_loss(gen_img)
        total_loss = target_loss + tv_loss.loss

        print(
            f"target loss: {target_loss.item()}, tv_loss: {tv_loss.loss.item()} epoch: {step[0]}"
        )
        step[0] += 1

        total_loss.backward()
        return total_loss

    optimizer.step(optim_step)

    save_image(gen_img.squeeze(0), "test-img.png")


if __name__ == "__main__":
    main()
