import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
from utils import calculate_BLEU


def main():
    pass


def caption(imgloader, encoder, decoder, device, **kwargs):
    decoder.eval()
    encoder.eval()


if __name__ == "__main__":
    main()
