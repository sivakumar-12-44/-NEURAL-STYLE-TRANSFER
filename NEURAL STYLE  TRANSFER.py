import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

def save_image(tensor, filename):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)
    print(f"✅ Image saved as {filename}")

class ContentLoss(nn.Module):
    def _init_(self, target):
        super(ContentLoss, self)._init_()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def _init_(self, target_feature):
        super(StyleLoss, self)._init_()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def _init_(self, mean, std):
        super(Normalization, self)._init_()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, x):
        return (x - self.mean) / self.std

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    norm = Normalization(norm_mean, norm_std).to(device)

    content_losses = []
    style_losses = []
    model = nn.Sequential(norm)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{name}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{name}", style_loss)
            style_losses.append(style_loss)

    # Trim off unused layers
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j+1]

    return model, style_losses, content_losses

def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img,
                       num_steps=300, style_weight=1e6, content_weight=1):
    
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, norm_mean, norm_std, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    print("🎨 Starting style transfer...")

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]} | Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


if _name_ == "_main_":
    content_img = image_loader("your_photo.jpg")
    style_img = image_loader("your_style.jpg")

    assert content_img.size() == style_img.size(), "Images must be the same size."

    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    imshow(output, title="Stylized Output")
    save_image(output, "stylized_output.jpg")
