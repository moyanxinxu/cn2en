from utils import Transformer
from config import hp
from dataset import loader
from tqdm import trange, tqdm
import mindspore as ms

from mindspore import nn, value_and_grad
from mindspore.experimental.optim import SGD, lr_scheduler
import numpy as np

train_part = loader("train", hp.batch_size, True)
test_part = loader("test", hp.batch_size, False)

try:
    ms.set_context(device_target=hp.device)
except:
    ms.set_context(device_target="CPU")

model = Transformer()

loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer = SGD(model.trainable_params(), lr=hp.lr)
scheduler = lr_scheduler.StepLR(optimizer, 2, 0.8)


def forward_fn(cn, en):
    output = model(cn, en[:, :-1])
    # [batch_size, max_length, en_vocab_size] --> [batch_size * max_length, en_vocab_size]
    # 一个max_length的序列,就代表一个词,这里用reshape不改变数据,只是改变了数据的形状,总共有batch_size * max_length个词
    output = output.reshape(-1, hp.en_vocab_size)
    en = en[:, 1:].reshape(-1)
    select = en != hp.special_tokens["<PAD>"]

    output = output[select]
    en = en[select]
    loss = loss_fn(output, en)
    return loss


backward_fn = value_and_grad(forward_fn, None, model.trainable_params())


def train_step(cn, en):
    loss, grads = backward_fn(cn, en)
    optimizer(grads)
    return loss


best = np.inf
for epoch in trange(hp.epoches, colour="green"):
    train_loss = 0
    model.set_train(True)
    for cn, en in train_part.create_tuple_iterator():
        train_loss += train_step(cn, en)

    test_loss = 0

    model.set_train(False)
    for cn, en in test_part.create_tuple_iterator():
        test_loss = forward_fn(cn, en)

    scheduler.step()
    last_lr = scheduler.get_last_lr()[0]

    if test_loss < best and train_loss < best:
        best = test_loss
        ms.save_checkpoint(model, hp.model_dir + f"{epoch}.ckpt")
        tqdm.write(
            f"Epoch:{epoch},Train Loss:{train_loss},Test Loss:{test_loss},lr:{last_lr},save model"
        )
    else:
        tqdm.write(
            f"Epoch:{epoch},Train Loss:{train_loss},Test Loss:{test_loss},lr:{last_lr}"
        )
