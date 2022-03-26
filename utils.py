from collections import Counter
import torch


def create_vocab2id(path_ann, ths_vocab):
    """Create vocabulary map

    Args:
        path_ann : path to annotation file
        ths_vocab (optional): threshold for vocabulary count
    """

    with open(path_ann, "r") as f:
        lst_ann = f.readlines()

    counter = Counter()
    counter.update(
        word for ann in lst_ann for word in ann.strip("\n").split("\t")[-1].split()
    )

    vocabs = [vocab for vocab in counter.keys() if counter[vocab] > ths_vocab]

    vocab2id = {vocab: id + 4 for id, vocab in enumerate(vocabs, 1)}
    vocab2id["<PAD>"] = 0
    vocab2id["<START>"] = len(vocab2id) + 1
    vocab2id["<END>"] = len(vocab2id) + 1
    vocab2id["<UNK>"] = len(vocab2id) + 1

    return vocab2id, vocabs


def decode_caption(caption, vocab2id):
    """Decode caption based on vocabulary map"""
    id2vocab = {id: vocab for vocab, id in vocab2id.items()}
    return " ".join(
        [id2vocab[id] for id in caption if not id2vocab[id].startswith("<")]
    )


def calculate_BLEU():
    """Calculate BLEU"""
