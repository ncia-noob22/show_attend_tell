import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.datasets import Flickr30k
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import create_voca2id


class CustomFlickr30k(Flickr30k):
    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        voca2id = create_voca2id(self.ann_file, ths_voca=5)
        len_caption_max = max(len(caption.split()) for caption in target)
        target = [
            torch.Tensor([voca2id.get(word, 0) for word in caption.split()])
            for caption in target
        ]
        target = torch.stack(
            [
                F.pad(caption, pad=(0, len_caption_max - caption.numel()))
                for caption in target
            ]
        )

        return img, target, len_caption_max


def get_dataloaders(dir_img, path_ann, num_batch, **kwargs):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = CustomFlickr30k(dir_img, path_ann, transform=transform)
    trainset, validset = data.random_split(  #! need to make fancier spliting code
        dataset, [int(len(dataset) * 0.7) + 1, int(len(dataset) * 0.3)]
    )

    trainloader = data.DataLoader(trainset, batch_size=num_batch, shuffle=True)
    validloader = data.DataLoader(validset, batch_size=num_batch, shuffle=True)

    return trainloader, validloader
