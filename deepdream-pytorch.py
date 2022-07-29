import torch
from torchvision.models import vgg19_bn, VGG19_BN_Weights
import utils
from PIL import Image
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image
from models import VGG19

device = {torch.has_cuda: "cuda", torch.has_mps: "mps"}.get(True, "cpu")

# training step
def train(gen_img, loss_model, optimizer):
    pass


def deepdream(args=None):
    features = VGG19(device=device)
    image = Image.open("temp.png")
    image = utils.preprocess(image).to(device)

    for step in range(200):
        ...


# generates what the model thinks a particular class is
def generate_class(args=None):
    loss_model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).eval().to(device)
    loss_model.requires_grad_(False)

    # image = Image.open("temp.png")
    # image = utils.preprocess(image).to(device)
    # print(torch.argmax(F.softmax(loss_model(image))))

    gen_img = torch.randn(size=(1, 3, 512, 512)).to(device) * 0.25 + 0.5

    optimizer = optim.SGD([gen_img], lr=1e-2)
    gen_img.requires_grad_(True)

    for step in range(4000):
        optimizer.zero_grad()

        target_loss = -loss_model(gen_img).squeeze(0)[282]
        tv_loss = utils.tv_loss(gen_img)
        total_loss = 1e2 * target_loss + 1 * tv_loss

        with torch.no_grad():
            if step % 500 == 0:
                save_image(gen_img.squeeze(0), "temp.png")
            if step % 100 == 0:
                print(
                    f"activation: {-target_loss.item()}, tv_loss: {tv_loss.item()} epoch: {step}"
                )

        total_loss.backward()
        optimizer.step()

    save_image(gen_img.squeeze(0), "test-img.png")


if __name__ == "__main__":
    deepdream()
