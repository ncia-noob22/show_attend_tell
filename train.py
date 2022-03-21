import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import os
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNEncoder, RNNDecoder
from dataset import get_dataloader

#!
dir_hub = "./"
num_epoch = 1
#!


def main():
    # basic settings
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.hub.set_dir(dir_hub)

    with open(f"{ljh_dir}/config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    pretrained_enc = config["pretrained_enc"]
    pretrained_dec = config["pretrained_dec"]
    num_epoch = config["num_epoch"]

    # load models
    encoder, decoder = CNNEncoder(**config).to(device), RNNDecoder(**config).to(device)
    if pretrained_enc:
        encoder.load_state_dict(pretrained_enc)
    if pretrained_dec:
        decoder.load_state_dict(pretrained_dec)

    # load data loaders
    trainloader, validloader = get_dataloader(**config)

    # load loss function
    loss = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epoch):

        # train models
        train()

        # validate models
        validate()


def train(
    trainloader,
    encoder,
    decoder,
    encoder_opt,
    decoder_opt,
    loss,
    epoch,
    device,
    **kwargs,
):
    decoder.train()
    encoder.train()

    mean_losses = []
    for imgs, caps in tqdm(trainloader):
        imgs, caps = imgs.to(device), caps.to(device)

        imgs = encoder(imgs)
        decoder(imgs, caps)  #! after decoder implementation

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        losses = loss()

        losses.backward()
        encoder_opt.step()
        decoder_opt.step()

    print(f"Mean loss is {sum(mean_losses) / len(mean_losses)} for {epoch + 1}th epoch")


def validate(validloader, encoder, decoder, loss, epoch, device, **kwargs):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        for imgs, caps in tqdm(validloader):
            imgs, caps = imgs.to(device), caps.to(device)


if __name__ == "__main__":
    main()
