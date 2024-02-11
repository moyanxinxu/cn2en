from config import hp
from mindspore import ops, nn
import numpy as np
import math


def mask_pad(data: ops.Tensor):
    # data:[batch_size, max_length]
    # 判断每个元素是否为PAD
    mask = data == hp.special_tokens["<PAD>"]
    # [batch_size, max_length] --> [batch_size, 1, max_length]
    mask = mask.unsqueeze(1)
    # [batch_size, 1, max_length] --> [batch_size, max_length, max_length]
    mask = mask.broadcast_to((-1, hp.max_length, hp.max_length))
    # [batch_size, max_length, max_length] -->  [batch_size, 1, max_length, max_length]
    return mask.unsqueeze(1)


def mask_tril(data: ops.Tensor):
    # data:[batch_size, max_length]
    # tril:[max_length, max_length]
    tril = 1 - ops.tril(ops.ones((hp.max_length, hp.max_length)))
    # tril:[1, max_length, max_length]
    tril = tril.unsqueeze(0)

    mask = data == hp.special_tokens["<PAD>"]
    # [batch_size, max_length] --> [batch_size, 1, max_length]
    mask = mask.unsqueeze(1)
    # [batch_size, 1, max_length] + [1, max_length, max_length] --> [batch_size, max_length, max_length]
    mask += tril
    mask = mask > 0
    # [batch_size, max_length, max_length] --> [batch_size, 1, max_length, max_length]
    mask = (mask == 1).unsqueeze(1)
    return mask


def attention(Q: ops.Tensor, K: ops.Tensor, V: ops.Tensor, mask: ops.Tensor):
    # 确保 mask.dtype: ms.bool_
    # Q, K, V:[batch_size, num_headers, max_length, embedding_size_per_header]
    # [batch_size, num_headers, max_length, embedding_size_per_header] * [batch_size, num_headers, embedding_size_per_header, max_length]
    # = [batch_size, num_headers, max_length, max_length]
    score = Q.matmul(K.permute(0, 1, 3, 2))
    # 除以每个根号下头的维度
    score /= math.sqrt(hp.embedding_size_per_header)
    # mask:[batch_size, num_headers, max_length, max_length]
    # 利用masked_fill()将所有1的位置设置为非常非常小的数
    score = score.masked_fill(mask, -1e9)
    score = ops.softmax(score)
    # [batch_size, num_headers, max_length, max_length] * [batch_size, num_headers, max_length, embedding_size_per_header]
    score = score.matmul(V)
    # [batch_size, num_headers, max_length, embedding_size_per_header] --> [batch_size, max_length, num_headers, embedding_size_per_header]
    score = score.permute(0, 2, 1, 3)
    # [batch_size, max_length, num_headers, embed在ding_size//num_headers] --> [batch_size, max_length, embedding_size]
    score = score.reshape(-1, hp.max_length, hp.embedding_size)
    return score


class MultiHead(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc_Q = nn.Dense(hp.embedding_size, hp.embedding_size)
        self.fc_K = nn.Dense(hp.embedding_size, hp.embedding_size)
        self.fc_V = nn.Dense(hp.embedding_size, hp.embedding_size)
        self.fc_O = nn.Dense(hp.embedding_size, hp.embedding_size)

        self.normal = nn.LayerNorm((hp.max_length, hp.embedding_size), 1, 1)
        self.dropout = nn.Dropout(p=0.1)

    def construct(self, Q, K, V, mask):
        Q_copy = Q.copy()
        Q, K, V = self.normal(Q), self.normal(K), self.normal(V)
        Q, K, V = self.fc_Q(Q), self.fc_K(K), self.fc_V(V)

        Q = Q.reshape(
            hp.batch_size, hp.max_length, hp.num_headers, hp.embedding_size_per_header
        ).permute(0, 2, 1, 3)

        K = K.reshape(
            hp.batch_size, hp.max_length, hp.num_headers, hp.embedding_size_per_header
        ).permute(0, 2, 1, 3)

        V = V.reshape(
            hp.batch_size, hp.max_length, hp.num_headers, hp.embedding_size_per_header
        ).permute(0, 2, 1, 3)

        score = attention(Q, K, V, mask)
        score = self.fc_O(score)
        score = self.dropout(score)
        return score + Q_copy


class Position_embedding(nn.Cell):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hp.embedding_size)
        self.position_mat = self.pos_embedding()

    def pos_embedding(self):
        # [max_length, embedding_size]
        mat = np.empty((hp.max_length, hp.embedding_size))

        # 获取每个位置的行索引,列索引
        # left: [max_length, embedding_size]
        # right: [max_length, embedding_size]
        left, right = np.indices((hp.max_length, hp.embedding_size))

        even = np.sin(left[:, ::2] / (1e4 ** (2 * right[:, ::2] / hp.embedding_size)))
        odd = np.cos(left[:, 1::2] / (1e4 ** (2 * right[:, 1::2] / hp.embedding_size)))

        mat[:, ::2] = even
        mat[:, 1::2] = odd
        # [max_length, embedding_size] --> [1, max_length, embedding_size]
        # 拓展一个维度,方便对于每个batch进行广播
        mat = np.expand_dims(mat, 0)
        # const_arg=True指定该Tensor作为网络输入时是常量
        return ops.Tensor(mat.astype(np.float32), const_arg=True)

    def construct(self, x):
        # [batch_size, max_length] --> [batch_size, max_length, embedding_size]
        x = self.embedding(x)
        # [batch_size, max_length, embedding_size] --> [batch_size, max_length, embedding_size]
        x += self.position_mat
        return x


class FullyConnectedOutput(nn.Cell):
    def __init__(self):
        super().__init__()
        self.normal = nn.LayerNorm((hp.max_length, hp.embedding_size), 1, 1)

        self.fc = nn.SequentialCell(
            nn.Dense(hp.embedding_size, 2 * hp.embedding_size),
            nn.ReLU(),
            nn.Dense(2 * hp.embedding_size, hp.embedding_size),
            nn.Dropout(p=0.1),
        )

    def construct(self, x):
        # [batch_size, max_length, embedding_size] --> [batch_size, max_length, embedding_size]
        norm_x = self.normal(x)
        # [batch_size, max_length, embedding_size] --> [batch_size, max_length, out_features]
        output = self.fc(norm_x)
        # [batch_size, max_length, out_features] --> [batch_size, max_length, out_features]
        return x + output


class EncoderLayer(nn.Cell):
    def __init__(self):
        super().__init__()

        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def construct(self, x, mask):
        # x:[batch_size, max_length, embedding_size]
        # mask:[batch_size, 1, max_length, max_length]
        # [batch_size, max_length, embedding_size] --> [batch_size, max_length, embedding_size]
        score = self.mh(x, x, x, mask)
        return self.fc(score)


class Encoder(nn.Cell):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderLayer()
        self.encoder2 = EncoderLayer()
        self.encoder3 = EncoderLayer()

    def construct(self, x, mask):
        # mask:[batch_size, 1, max_length, max_length]
        # x: [batch_size, max_length, embedding_size]
        x = self.encoder1(x, mask)
        x = self.encoder2(x, mask)
        x = self.encoder3(x, mask)
        return x


class DecoderLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()
        self.fc = FullyConnectedOutput()

    def construct(self, x, y, mask_pad, mask_tril):
        # mask_tril 用于计算y的自注意力
        y = self.mh1(y, y, y, mask_tril)
        # mask_pad 用于计算y与x之间的注意力
        y = self.mh2(y, x, x, mask_pad)
        return self.fc(y)


class Decoder(nn.Cell):
    def __init__(self):
        super().__init__()
        self.decoder1 = DecoderLayer()
        self.decoder2 = DecoderLayer()
        self.decoder3 = DecoderLayer()

    def construct(self, x, y, mask_pad, mask_tril):
        y = self.decoder1(x, y, mask_pad, mask_tril)
        y = self.decoder2(x, y, mask_pad, mask_tril)
        y = self.decoder3(x, y, mask_pad, mask_tril)
        return y


class Transformer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.embed_x = Position_embedding(hp.cn_vocab_size)
        self.embed_y = Position_embedding(hp.en_vocab_size)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.fc_out = nn.Dense(hp.embedding_size, hp.en_vocab_size)

    def construct(self, x, y):
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)
        x = self.embed_x(x)
        y = self.embed_y(y)
        x = self.encoder(x, mask_pad_x)
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        y = self.fc_out(y)
        return y


def predict(x, model):
    sos = [hp.special_tokens["<SOS>"]]
    pad = [hp.special_tokens["<PAD>"]]
    # x:[1, max_length], model:Transformer
    # mask_pad_x:[1, 1, max_length, max_length]
    # 第一个维度留给batch_size，第二个维度留给head，第三个维度留给query，第四个维度留给key
    mask_pad_x = mask_pad(x)
    # [1, max_length] --> [1, max_length, embedding_size]
    x = model.embed_x(x)
    # [1, max_length, embedding_size] --> [1, max_length, embedding_size]
    x = model.encoder(x, mask_pad_x)

    # target:[max_length]
    target = sos + pad * (hp.max_length - 1)
    # target:[max_length] --> [1, max_length]
    target = ops.Tensor(np.array(target).astype(np.int32)).unsqueeze(0)

    # 遍历预先定义的最大长度
    for i in range(hp.max_length - 1):
        # 这里的target每一个迭代都被更新,当前的target是当前所有预测的词,其余全是PAD,下一轮的target将是上一轮
        # 所有预测的词并按顺序替换掉一个PAD后生成的序列
        # y: [1, max_length]
        y = target
        # [1, max_length] --> [1, 1, max_length, max_length]
        mask_tril_y = mask_tril(y)
        # 对y进行位置编码
        # [1, max_length] --> [1, max_length, embedding_size]
        y = model.embed_y(y)
        # 不断解码
        # [1, max_length, embedding_size] --> [1, max_length, embedding_size]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        # [1, max_length, embedding_size] --> [1, max_length, en_vocab_size]
        out = model.fc_out(y)
        # [1, max_length, en_vocab_size] --> [1, en_vocab_size]
        # 取出当前预测的词
        out = out[:, i, :]
        out = out.argmax(-1)
        target[:, i + 1] = out
    return target
