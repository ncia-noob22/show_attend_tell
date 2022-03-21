import torch.nn as nn
from torchvision.models import vgg11

# !
size_enc = 14
import torch

dir_hub = "./"
torch.hub.set_dir(dir_hub)
# !


class CNNEncoder(nn.Module):
    """Encoder based on pretrained VGGnet"""

    def __init__(self, size_enc, **kwargs):
        super().__init__()
        vgg = vgg11(pretrained=True)
        cnn_layers = list(vgg.children())[0].append(
            nn.AdaptiveAvgPool2d(output_size=size_enc)
        )
        self.encoder = nn.Sequential(*cnn_layers)

    def forward(self, img):
        # N ✕ (...) -> N ✕ 512 ✕ 196 -> N ✕ 196 ✕ 512
        return self.encoder(img).flatten(-2).permute(0, 2, 1)


class SoftAttention(nn.Module):
    """Soft attention based on Bahdanau attention"""

    def __init__(self, dim_enc, dim_dec, dim_attn):
        super().__init__()
        self.attn_img = nn.Linear(dim_enc, dim_attn)
        self.attn_txt = nn.Linear(dim_dec, dim_attn)
        self.tanh = nn.Tanh()

        self.attn_last = nn.Linear(dim_attn, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded_img, previous_txt):
        score = self.attn_last(
            self.tanh(self.attn_img(encoded_img) + self.attn_txt(previous_txt))
        )
        attention = self.softmax(score) * encoded_img
        return attention


class HardAttention(nn.Module):
    """Hard attention with Monte-Carlo sampling"""

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class RNNDecoder(nn.Module):
    """LSTM-based decoder"""

    def __init__(
        self, dim_enc, dim_dec, dim_attn, dim_emb, size_vocab, type_attn, **kwargs
    ):
        super().__init__()
        self.dim_enc = dim_enc
        self.dim_dec = dim_dec

        if type_attn.lower() == "soft":
            self.attn = SoftAttention(dim_enc, dim_dec, dim_attn)
        elif type_attn.lower() == "hard":
            self.attn = HardAttention()

        self.decoder = nn.LSTM(dim_enc, dim_dec)

        self.embedding = nn.Embedding(size_vocab, dim_emb)

    def init_rnn(self, encoded_img):
        init_c = nn.Linear(self.dim_enc, self.dim_dec)
        init_h = nn.Linear(self.dim_enc, self.dim_dec)

        c = init_c(encoded_img.mean(dim=1))
        h = init_h(encoded_img.mean(dim=1))
        return c, h

    def forward(self, encoded_img, encoded_txt):
        emb_txt = self.embedding(encoded_txt)

        c, h = self.init_rnn(encoded_img)

        # ~ need to study and align with dataset


if __name__ == "__main__":
    import yaml
    import torch
    import torchsummary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    encoder = CNNEncoder(**config).to(device)
    decoder = RNNDecoder(**config).to(device)
    print(torchsummary.summary(model, (3, 448, 448), device=device.split(":")[0]))
