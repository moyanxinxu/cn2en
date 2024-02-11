from config import hp
from mindspore.dataset.text import Vocab
from mindspore.dataset import GeneratorDataset


def get_vocab_size(txt):
    data = load_from_txt(txt)
    vocab = data2dic(data)
    return len(vocab)


def load_from_txt(txt):
    with open(txt, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data


def data2dic(data):
    word2id = hp.special_tokens
    for line in data:
        tokens = line.split()
        for token in tokens:
            if token not in word2id:
                word2id[token] = len(word2id)
    return word2id


def parser(sentence, max_length):
    sentence = ["<SOS>"] + sentence.split() + ["<EOS>"] + ["<PAD>"] * max_length
    return sentence[:max_length]


class cn2en:
    def __init__(self, split) -> None:
        self.split = split

        if split == "train":
            self.cn = load_from_txt(hp.cn_train)
            self.en = load_from_txt(hp.en_train)
            self.cn_vocab = Vocab.from_dict(data2dic(self.cn))
            self.en_vocab = Vocab.from_dict(data2dic(self.en))
        else:
            self.cn = load_from_txt(hp.cn_test)
            self.en = load_from_txt(hp.en_test)
            self.cn_vocab = Vocab.from_dict(data2dic(self.cn))
            self.en_vocab = Vocab.from_dict(data2dic(self.cn))

    def __len__(self) -> int:
        return len(self.cn)

    def __getitem__(self, idx):
        cn = self.cn[idx]
        en = self.en[idx]
        cn = parser(cn, hp.max_length)
        en = parser(en, hp.max_length + 1)
        cn = self.cn_vocab.tokens_to_ids(cn)
        en = self.en_vocab.tokens_to_ids(en)
        return cn, en


def loader(split, batch_size, shuffle):
    dataset = cn2en(split)
    if split == "train":
        shuffle = True
    else:
        shuffle = False

    dataset = GeneratorDataset(dataset, column_names=hp.columns, shuffle=shuffle)
    return dataset.batch(batch_size, drop_remainder=True)
