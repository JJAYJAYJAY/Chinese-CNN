# coding: UTF-8
import torch
import pickle as pkl


UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_single_text(config, ues_word, text, pad_size=32):
    '''
    处理单个文本字符串的函数，返回值格式与原始 build_dataset 函数一致。

    参数:
    - config: 包含配置信息的对象，比如词汇表的路径等。
    - ues_word: 布尔值，指示是否使用单词级别的分词。
    - text: 需要处理的单个文本字符串。
    - pad_size: 整数，指定序列的固定长度。

    返回:
    - kind: 返回值格式为 [([...], 0, seq_len)]，模拟原函数中的数据集结构。
    '''
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    vocab = pkl.load(open(config.vocab_path, 'rb'))

    token = tokenizer(text)
    seq_len = len(token)
    words_line = []

    # 多切少补
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size

    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK, 0)))

    kind = [(words_line, 0, seq_len)]

    return vocab, kind


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        # 使用batch_size 防止 n_batches == 0
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

