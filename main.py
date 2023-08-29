import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import tqdm
from transformer import Transformer


class dictionary:
    def __init__(self, pad_id: int, pad_letter: str, dic: dict) -> None:
        self.dic = dic
        self.pad_id = pad_id
        self.pad_letter = pad_letter
        self.rev_dic = dict(zip(dic.values(), dic.keys()))


class MyModel(nn.Module):
    def __init__(self, layer_num, vocab_size, embed_dim, head, hidden_size) -> None:
        super().__init__()
        self.head = head
        self.transformer = Transformer(
            layer_num, embed_dim, head, hidden_size, vocab_size
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.transformer(src, tgt, src_mask, tgt_mask)


class myDataset(Dataset):
    def __init__(self, length, max_num_len) -> None:
        super().__init__()
        self.length = length
        self.max_num = max_num_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        a = random.randint(1, self.max_num)
        b = random.randint(1, self.max_num)
        a = random.randint(0, int(pow(10, a)))
        b = random.randint(0, int(pow(10, b)))
        return str(a) + "+" + str(b) + "=", "<" + str(a + b) + ">"  # 方便后面只在答案里预测


word2num = {"-": 0, "+": 11, "=": 12, ".": 13, "<": 14, ">": 15}
for i in range(10):
    word2num[str(i)] = i + 1

dic = dictionary(pad_id=0, pad_letter="-", dic=word2num)


def collate_fn(data):
    x = []
    y = []
    len_x = []
    len_y = []
    max_len = max([len(i[0]) for i in data])
    max_len2 = max([len(i[1]) for i in data])
    for unit in data:
        unit_x = [word2num[i] for i in unit[0]]
        unit_y = [word2num[i] for i in unit[1]]
        length = len(unit[0])
        length2 = len(unit[1])
        x.append(torch.tensor(unit_x))
        y.append(torch.tensor(unit_y))
        len_x.append(length * [False] + (max_len - length) * [True])
        len_y.append(length2 * [False] + (max_len2 - length2) * [True])
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    len_x = torch.tensor(len_x)
    len_y = torch.tensor(len_y)
    # 将lengths([B,max_len])转化为[B,max_len,1], 然后广播成[B,max_len,max_len]
    len_x = len_x.unsqueeze(-1).repeat(1, 1, len_x.shape[-1])
    len_x = torch.concat([len_x], dim=-1)
    return x, y, len_x, len_y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = myDataset(20000, 20)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

model = MyModel(6, len(word2num), 128, 8, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


# 初始化模型线性层的权重
def originate(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


model.apply(originate)


def train(it):
    model.to(device)
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))  # 进度条
    for i, (src, tgt, len_x, len_y) in pbar:
        optimizer.zero_grad()
        src_mask = len_x
        tgt_mask = torch.triu(
            torch.ones(tgt.shape[-1] - 1, tgt.shape[-1] - 1), diagonal=1
        )  # 生成上三角矩阵, 用于屏蔽未来的信息
        tgt_mask = tgt_mask.unsqueeze(0).repeat(
            tgt.shape[0], 1, 1
        )  # [B,tgt_len-1,tgt_len-1]
        # 将mask转换为bool类型
        tgt_mask = tgt_mask.type(torch.bool)
        src_mask = src_mask.type(torch.bool)
        output = model(
            src.to(device),
            tgt[..., :-1].to(device),  # 位移一位, 用于预测下一个字符
            src_mask.to(device),
            tgt_mask.to(device),
        )
        tgt = tgt[..., 1:].to(device)  # 答案也位移一位
        loss = loss_fn(output.reshape(-1, len(word2num)), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        # 计算准确率
        output = output.argmax(dim=-1)
        total = (tgt != 0).sum()
        correct = (output == tgt).sum() - (tgt == 0).sum()
        acc = correct / total
        total_number = tgt.shape[0]
        correct_number = [
            1 if tgt[j].equal(output[j]) else 0 for j in range(total_number)
        ]
        correct_number = sum(correct_number)
        acc_number = correct_number / total_number
        pbar.set_description(
            "iter: %d, loss: %.4f, acc: %.4f, acc_number: %.4f"
            % (it, loss.item(), acc.item(), acc_number)
        )  # 更新进度条

    torch.save(model.state_dict(), "model.pth")

    # 测试
    x = src[0].tolist()
    predict = output[0].tolist()
    y = tgt[0].tolist()
    x = [dic.rev_dic[i] for i in x]
    predict = [dic.rev_dic[i] for i in predict]
    y = [dic.rev_dic[i] for i in y]
    x = "".join(x)
    predict = "".join(predict)
    y = "".join(y)
    print(x, y, predict)


def test(a, b, to_print=False):
    model.to("cpu")
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    x = str(a) + "+" + str(b) + "="
    x = [word2num[i] for i in x]
    x = torch.tensor(x).unsqueeze(0)
    y = ["<"]
    y = [word2num[i] for i in y]
    src_mask = None
    tgt_mask = None
    cnt = 0
    while (len(y) == 0 or y[-1] != word2num[">"]) and cnt < 100:
        cnt += 1
        output = model(
            x, torch.tensor(y, dtype=torch.int).unsqueeze(0), src_mask, tgt_mask
        )
        output = output.argmax(dim=-1)
        y.append(output[0, -1].item())
    y = "".join([dic.rev_dic[i] for i in y[1:-1]])
    real_ans = str(int(a) + int(b))
    if to_print:
        print(y)
        print(real_ans)
    if y == real_ans:
        return True


model.load_state_dict(torch.load("model.pth"))
for i in range(100):
    train(i)

# 测试
total = 0
correct = 0
pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))  # 进度条
for i, (x, y, len_x, len_y) in pbar:
    x = x[0].tolist()
    x = [dic.rev_dic[i] for i in x]
    x = "".join(x)
    a, b = x.split("+")
    b, c = b.split("=")
    total += 1
    correct += 1 if test(a, b) else 0
    pbar.set_description("acc: %.4f" % (correct / total))  # 更新进度条
print(correct / total)
