import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNEncoder, RNNDecoder
from dataset import get_dataloader
from train import train

#!
num_epoch = 100
#!


def main():
    # basic settings
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.hub.set_dir("./tmp/checkpoints/")

    with open(f"{ljh_dir}/config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    path_pretrained = config["path_pretrained"]
    only_train = config["only_train"]
    num_epoch = config["num_epoch"]

    # load data loaders
    trainloader, validloader = get_dataloader(**config)

    # load models
    encoder, decoder = CNNEncoder(**config).to(device), RNNDecoder(**config).to(device)
    if path_pretrained:
        encoder.load_state_dict(path_pretrained)
        decoder.load_state_dict(path_pretrained)

    # load loss function
    loss = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epoch):

        # train models
        train()

        # validate models


if __name__ == "__main__":
    main()
