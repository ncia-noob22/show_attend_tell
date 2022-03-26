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

        # for Doubly Stochastic Attention
        self.f_beta = nn.Linear(dim_dec, dim_enc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_img, hidden_dec):
        score = self.attn_last(
            self.tanh(self.attn_img(encoded_img) + self.attn_txt(hidden_dec))
        )
        attn = self.softmax(score) * encoded_img

        # Doubly Stochastic Attention
        gate = self.sigmoid(self.f_beta(hidden_dec))

        return gate * attn


class HardAttention(nn.Module):  #! Not yet implementing
    """Hard attention with Monte-Carlo sampling"""

    def __init__(self, dim_enc, dim_dec, dim_attn):
        super().__init__()

    def forward(self, encoded_img, hidden_dec):
        pass


class RNNDecoder(nn.Module):
    """LSTM-based decoder"""

    def __init__(
        self, dim_enc, dim_dec, dim_attn, dim_emb, size_vocab, type_attn, **kwargs
    ):
        super().__init__()
        self.dim_enc = dim_enc
        self.dim_dec = dim_dec
        self.size_vocab = size_vocab

        self.embedding = nn.Embedding(size_vocab, dim_emb)

        if type_attn.lower() == "soft":
            self.attn = SoftAttention(dim_enc, dim_dec, dim_attn)
        elif type_attn.lower() == "hard":
            self.attn = HardAttention(dim_enc, dim_dec, dim_attn)

        self.rnn = nn.LSTMCell(dim_enc + dim_emb, dim_dec)

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(dim_dec, size_vocab)

    def init_rnn(self, encoded_img):
        init_c = nn.Linear(self.dim_enc, self.dim_dec)
        init_h = nn.Linear(self.dim_enc, self.dim_dec)

        cell = init_c(encoded_img.mean(dim=1))
        hidden = init_h(encoded_img.mean(dim=1))
        return cell, hidden

    def forward(self, encoded_img, encoded_txt, len_caption):
        emb_txt = self.embedding(encoded_txt)  # N ✕ len_caption_max ✕ dim_emb

        cell, hidden = self.init_rnn(encoded_img)

        size_batch = encoded_img.shape[0]
        len_decode = len_caption.max()

        preds = torch.zeros(size_batch, len_decode, self.size_vocab).to(device)

        for t in range(len_decode):
            size_batch_t = sum([l > t for l in len_decode])
            attn = self.attn(encoded_img[:size_batch_t], hidden[:size_batch_t])

            hidden, cell = self.rnn(
                torch.cat([emb_txt[:size_batch_t, t, :], attn], dim=1),
                (hidden[:size_batch_t], cell[:size_batch_t]),
            )
            preds[:size_batch_t, t, :] = self.fc(self.dropout(hidden))

        return preds, len_decode


if __name__ == "__main__":
    import yaml
    import torch
    import torchsummary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    encoder = CNNEncoder(**config).to(device)
    decoder = RNNDecoder(**config).to(device)
    model = decoder(encoder())

    print(torchsummary.summary(model, (3, 448, 448), device=device.split(":")[0]))
