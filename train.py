from tqdm import tqdm
import torch


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

        losses.backward()
        opt.step()

    print(f"Mean loss is {sum(mean_losses) / len(mean_losses)} for {epoch + 1}th epoch")


def validate(validloader, encoder, decoder, loss, epoch, device, **kwargs):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        for imgs, caps in tqdm(validloader):
            imgs, caps = imgs.to(device), caps.to(device)
