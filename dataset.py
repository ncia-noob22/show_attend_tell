from PIL import Image
import torch
from torchvision.datasets import Flickr30k
import torch.utils.data as data
import torchvision.transforms as transforms


class CustomFlickr30k:
    """Custom Flickr30k dataset"""


def get_dataloaders(dir_img, dir_ann, batch_size, only_train, **kwargs):
    transform = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    )

    if only_train:
        trainset = CustomFlickr30k(
            dir_img, dir_ann, image_set="train", transform=transform
        )
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        return trainloader, None

    else:
        trainset = CustomFlickr30k(
            dir_img, dir_ann, image_set="train", transform=transform
        )
        validset = CustomFlickr30k(
            dir_img, dir_ann, image_set="val", transform=transform
        )

        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validloader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

        return trainloader, validloader
