import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import yaml
    import torch
    import torchsummary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    model = YOLOv1(**config).to(device)
    print(torchsummary.summary(model, (3, 448, 448), device=device.split(":")[0]))
