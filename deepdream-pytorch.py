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
    image = Image.open("dog.jpg")
    image = utils.preprocess(image).to(device)

    gen_img = image.clone()
    optimizer = optim.SGD([gen_img], lr=1e-2)
    gen_img.requires_grad_(True)

    num_steps = 400
    for step in range(num_steps):
        optimizer.zero_grad()

        activations = features(gen_img)
        with torch.no_grad():
            if step % 50 == 0:
                print(
                    f"activations: {[a.item() for a in activations]}  [{step:>5d}/{num_steps:>5d}]"
                )
                save_image(utils.denormalize(gen_img).squeeze(0), "temp.png")

        # per pixel loss l2 regularizer
        per_pixel_loss = F.mse_loss(gen_img, image)

        loss = (
            -1e4 * sum(activations)
            + 1e-2 * utils.tv_loss(gen_img)
            + 1e2 * per_pixel_loss
        )
        loss.backward()
        optimizer.step()

    save_image(utils.denormalize(gen_img).squeeze(0), "test-image.png")


# generates what the model thinks a particular class is
def generate_class(args=None):
    loss_model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).eval().to(device)
    loss_model.requires_grad_(False)

    # image = Image.open("temp.png")
    # image = utils.preprocess(image).to(device)
    # print(torch.argmax(F.softmax(loss_model(image))))

    gen_img = torch.randn(size=(1, 3, 512, 512)).to(device) * 0.25 + 0.5

    optimizer = optim.Adam([gen_img], lr=1e-2)
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
