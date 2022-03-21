from collections import Counter
import torch


def create_voca2id(path_ann, ths_voca):
    """Create vocabulary map

    Args:
        path_ann : path to annotation file
        ths_voca (optional): threshold for vocabulary count
    """

    with open(path_ann, "r") as f:
        lst_ann = f.readlines()

    counter = Counter()
    counter.update(
        word for ann in lst_ann for word in ann.strip("\n").split("\t")[-1].split()
    )

    vocas = [voca for voca in counter.keys() if counter[voca] > ths_voca]

    voca2id = {voca: id + 4 for id, voca in enumerate(vocas)}
    voca2id["<START>"] = len(voca2id) + 1
    voca2id["<END>"] = len(voca2id) + 1
    voca2id["<UNK>"] = len(voca2id) + 1
    voca2id["<PAD>"] = 0

    return voca2id, vocas


def decode_caption(caption, voca2id):
    """Decode caption based on vocabulary map"""
    id2voca = {id: voca for voca, id in voca2id.items()}
    return " ".join([id2voca[id] for id in caption if not id2voca[id].startswith("<")])


def calculate_BLEU():
    """Calculate BLEU"""
