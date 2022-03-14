import torch.nn as nn
import torch.functional as F
from torchvision.models import vgg11

# !
import torch

torch.hub.set_dir("./tmp/")
# !


class CNNEncoder(nn.Module):
    """Encoder based on pretrained VGGnet"""

    def __init__(self):
        super().__init__()
        vgg = vgg11(pretrained=True)
        cnn_layers = list(vgg.children())[:1] + [nn.AdaptiveAvgPool2d(output_size=14)]
        self.encoder = nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.encoder(x)


class SoftAttention(nn.Module):
    def __init__(self, dim_enc, dim_dec, dim_att):
        super().__init__()
        self.attn_enc = nn.Linear(dim_enc, dim_att)
        self.attn_dec = nn.Linear(dim_dec, dim_att)
        self.relu = nn.ReLU()

        self.attn_last = nn.Linear(dim_att, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, out_enc, hid_dec):
        att_a = self.attn_enc(out_enc)
        att_d = self.attn_dec(hid_dec)

        # self.attn_last(self.relu(att_a + att_d))


class HardAttention(nn.Module):
    def __init__(self):
        super().__init__()


class RNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    import yaml
    import torch
    import torchsummary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    model = CNNEncoder(**config).to(device)
    print(torchsummary.summary(model, (3, 448, 448), device=device.split(":")[0]))
