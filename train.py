# -*- coding:utf-8 -*-
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from collections import Counter
import numpy as np
import random
import math

import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import configargparse

from data import WordEmbeddingDataset
from model import EmbeddingModel


# 设定超参数
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=100,
                        help='number of negative samples')
    parser.add_argument('--C', type=int, default=3,
                        help='nearby words threshold (window size)')
    parser.add_argument('--t', type=float, default=1e-5,
                        help='Subsampling threshold')
    parser.add_argument('--NUM_EPOCHS', type=int, default=5,
                        help='the vocabulary size')
    parser.add_argument('--MAX_VOCAB_SIZE', type=int, default=10000,
                        help='number of negative samples')
    parser.add_argument('--BATCH_SIZE', type=int, default=128,
                        help='batch size')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.2,
                        help='the initial learning rate')
    parser.add_argument('--EMBEDDING_SIZE', type=int, default=100,
                        help='Number of word embedding features')
    parser.add_argument('--LOG_FILE', type=str, default="logs/word-embedding.log",
                        help='log file name')
    parser.add_argument('--TXT_NAME', type=str, default="data/zh.txt",
                        help='log file name')

    return parser


# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
if True:
    random.seed(53113)
    np.random.seed(53113)
    torch.manual_seed(53113)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(53113)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()


def read_txt(txt_name):
    """
    从文本文件中读取所有的文字，通过这些文本创建一个vocabulary
    由于单词数量可能太大，我们只选取最常见的MAX_VOCAB_SIZE个单词
    我们添加一个UNK单词表示所有不常见的单词
    我们需要记录单词到index的mapping，以及index到单词的mapping，单词的count，单词的(normalized) frequency，以及单词总数。
    """
    if 'zh' in txt_name:
        with open(txt_name, "r", encoding='utf-8') as fin:
            text = fin.read()
    else:
        with open(txt_name, "r") as fin:
            text = fin.read()

    text = [w for w in word_tokenize(text.lower())]
    # 选择常用的 MAX_VOCAB_SIZE 个单词, 后面所有不常用的词统一用unk表示
    vocab = dict(Counter(text).most_common(args.MAX_VOCAB_SIZE - 1))
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    idx_to_word = [word for word in vocab.keys()]
    word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
    return text, vocab, idx_to_word, word_to_idx


def draw_model():
    text, vocab, idx_to_word, word_to_idx = read_txt(txt_name=args.TXT_NAME)

    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_del_freqs = max(0., 1 - math.sqrt(args.t / word_freqs))
    word_freqs = word_freqs ** (3. / 4.)
    word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
    VOCAB_SIZE = len(idx_to_word)

    dataset = WordEmbeddingDataset(args, text, word_to_idx, idx_to_word, word_freqs, word_del_freqs, VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=1)

    input_labels, pos_labels, neg_labels = next(iter(dataloader))
    model = EmbeddingModel(VOCAB_SIZE, args.EMBEDDING_SIZE).to(device)

    input_labels = input_labels.long().to(device)
    pos_labels = pos_labels.long().to(device)
    neg_labels = neg_labels.long().to(device)

    writer = SummaryWriter()
    writer.add_graph(model, (input_labels, pos_labels, neg_labels))


def train():
    print()
    print("read txt from " + args.TXT_NAME)
    print("log file: " + args.LOG_FILE)
    print()
    text, vocab, idx_to_word, word_to_idx = read_txt(txt_name=args.TXT_NAME)

    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_del_freqs = 1 - np.sqrt(args.t / word_freqs)
    word_del_freqs[word_del_freqs < 0] = 0
    word_freqs = word_freqs ** (3. / 4.)
    word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
    VOCAB_SIZE = len(idx_to_word)
    print(VOCAB_SIZE)

    dataset = WordEmbeddingDataset(args, text, word_to_idx, idx_to_word, word_freqs, word_del_freqs, VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=1)

    model = EmbeddingModel(VOCAB_SIZE, args.EMBEDDING_SIZE).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.LEARNING_RATE)

    writer = SummaryWriter(os.path.join('logs', 'summary'))
    if not os.path.exists(os.path.join('logs', 'summary')):
        os.mkdir(os.path.join('logs', 'summary'))

    print('begin training ...')
    print("-----------------------------------------------------------------------------------")
    cnt = 0
    for epoch in range(args.NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            cnt += 1
            input_labels = input_labels.long().to(device)
            pos_labels = pos_labels.long().to(device)
            neg_labels = neg_labels.long().to(device)

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(args.LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {}\n".format(epoch, i, loss.item()))
                    print("epoch: {}, iter: {}, loss: {}".format(epoch, i, loss.item()))
                writer.add_scalars('loss', {'loss': loss}, cnt)

        embedding_weights = model.input_embeddings()
        np.save("logs/embedding-{}".format(args.EMBEDDING_SIZE), embedding_weights)
        torch.save(model.state_dict(), "logs/embedding-{}.th".format(args.EMBEDDING_SIZE))
        writer.close()
        print('---------------------------------------------------------------------------------')


def draw_PCA(word_list, Ch=False, embed_matrix_path=None, embedding_weights=None):
    if embedding_weights is None:
        embedding_weights = np.load(embed_matrix_path)

    text, vocab, idx_to_word, word_to_idx = read_txt(txt_name=args.TXT_NAME)
    word_embeddings = []
    for word in word_list:
        index = word_to_idx[word]
        embedding = embedding_weights[index]
        word_embeddings.append(embedding)

    word_embeddings = np.array(word_embeddings)
    print(word_embeddings.shape)
    pca = PCA(n_components=2)
    pca = pca.fit(word_embeddings)
    res = pca.transform(word_embeddings)
    print(res.shape)

    # 可视化
    plt.figure()
    if Ch:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    for i, word in enumerate(word_list):
        plt.scatter(res[i, 0], res[i, 1], c='red')
        plt.text(res[i, 0], res[i, 1], word)

    plt.title("PCA of word embedding") 
    plt.show()


def find_nearest(word, embed_matrix_path=None, embedding_weights=None):
    if embedding_weights is None:
        embedding_weights = np.load(embed_matrix_path)

    text, vocab, idx_to_word, word_to_idx = read_txt(txt_name=args.TXT_NAME)
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[1:4]]


if __name__ == '__main__':
    # global args
    if not os.path.exists('./logs'):
        os.mkdir("./logs")

    parsers = config_parser()
    args = parsers.parse_args()
    # draw_model()

    train()

    # neighbor
    if 'zh' in args.TXT_NAME:
        for word in ["自由", "好", "与", "一定", "发生", "是", "为", "人民", "政府", "中国"]:
            print(word, find_nearest(word, embed_matrix_path='./logs/embedding-100.npy'))
    else:
        for word in ["a", "an", "the", "good", "better", "best", "china", "river", "is", "are"]:
            print(word, find_nearest(word, embed_matrix_path='./logs/embedding-100.npy'))

    # PCA
    if 'zh' in args.TXT_NAME:
        word_list = ["自由", "好", "与", "一定", "发生", "是", "为", "人民", "政府", "中国"]
        draw_PCA(word_list, embed_matrix_path='./logs/embedding-100.npy', Ch=True)
    else:
        word_list = ["a", "an", "the", "good", "better", "best", "china", "river", "is", "are"]
        draw_PCA(word_list, embed_matrix_path='./logs/embedding-100.npy')
