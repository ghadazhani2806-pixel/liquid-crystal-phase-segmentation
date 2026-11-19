
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

CHECKPOINT_DIR = Path("checkpoints")

# ==========================================
# 2. U-NET
# ==========================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = DoubleConv(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# ==========================================
# 3. RESNET18-U-NET
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet18UNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.center = ConvBlock(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec4 = ConvBlock(128+256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = ConvBlock(64+128, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec2 = ConvBlock(64+64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(32+64, 32)

        self.up0 = nn.ConvTranspose2d(32, 32, 2, 2)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        z = self.center(x4)
        z = self.dec4(torch.cat([self.up4(z), x3], dim=1))
        z = self.dec3(torch.cat([self.up3(z), x2], dim=1))
        z = self.dec2(torch.cat([self.up2(z), x1], dim=1))
        z = self.dec1(torch.cat([self.up1(z), x0], dim=1))
        z = self.up0(z)

        return self.out(z)


# ==========================================
# 4. LOAD TRAINED MODELS
# ==========================================

def load_model(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


unet = load_model(
    UNet(base=32),
    CHECKPOINT_DIR / "unet_baseline_best.pth"
)

resunet = load_model(
    ResNet18UNet(pretrained=False),
    CHECKPOINT_DIR / "resnet18_unet_best.pth"
)

print("Les deux modèles ont été chargés avec succès.")
print("Device :", DEVICE)


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_upload(pil_img):
    img = np.array(pil_img.convert("RGB"))

    resized = Image.fromarray(img).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std

    tensor = torch.from_numpy(
        arr.transpose(2, 0, 1).copy()
    ).float()

    return tensor, np.asarray(resized)


# ==========================================
# 6. PREDICTION
# ==========================================

@torch.no_grad()
def predict_upload(pil_img, model_choice):

    if pil_img is None:
        return None, None, None, "Veuillez importer une image."

    model = unet if model_choice == "U-Net" else resunet

    tensor, resized = preprocess_upload(pil_img)

    pred = (
        model(tensor.unsqueeze(0).to(DEVICE))
        .argmax(1)
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    mask_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    mask_rgb[pred == 1] = [255, 0, 0]

    overlay = (
        0.6 * resized + 0.4 * mask_rgb
    ).clip(0, 255).astype(np.uint8)

    percent = 100 * pred.mean()

    analysis = f"Proportion estimée de Cr : {percent:.2f} %"

    return resized, mask_rgb, overlay, analysis


# ==========================================
# 7. GRADIO INTERFACE
# ==========================================

demo = gr.Interface(
    fn=predict_upload,
    inputs=[
        gr.Image(
            type="pil",
            label="Image de microscopie"
        ),
        gr.Radio(
            ["U-Net", "ResNet18-U-Net"],
            value="ResNet18-U-Net",
            label="Modèle"
        )
    ],
    outputs=[
        gr.Image(label="Image redimensionnée"),
        gr.Image(label="Masque prédit (Cr en rouge)"),
        gr.Image(label="Superposition"),
        gr.Textbox(label="Analyse")
    ],
    title="Segmentation des phases du cristal liquide 9BA4",
    description=(
        "Détection exploratoire des régions cristallines (Cr) "
        "et de la phase SmC."
    )
)


if __name__ == "__main__":
    demo.launch(share=True)
