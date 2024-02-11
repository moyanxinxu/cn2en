class hp:
    cn_train = "../data/cn.txt"
    cn_test = "../data/cn.test.txt"
    en_train = "../data/en.txt"
    en_test = "../data/en.test.txt"
    model_dir = "../model/"
    columns = ["cn", "en"]
    max_length = 512
    batch_size = 32
    special_tokens = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2}
    num_headers = 8
    embedding_size = 64
    cn_vocab_size = 13294
    en_vocab_size = 24966
    embedding_size_per_header = embedding_size // num_headers
    lr = 0.1
    epoches = 10
    device = "CPU"
    # device = "GPU"
    # device = 'Ascend'
