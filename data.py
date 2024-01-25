# -*- coding:utf-8 -*-
import torch
from torch.utils.data import Dataset


class WordEmbeddingDataset(Dataset):
    def __init__(self, args, text, word_to_idx, idx_to_word, word_freqs, word_del_freqs, VOCAB_SIZE, do_subsampling=False):
        """
        text: a list of words, all text from the training dataset
        word_to_idx: the dictionary from word to idx
        idx_to_word: idx to word mapping
        word_freq: the frequency of each word
        word_counts: the word counts
        """
        super(WordEmbeddingDataset, self).__init__()
        self.C = args.C
        self.K = args.K
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]  # 这个是text 的每个单词在词典word_to_idx 中的位置
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_del_freqs = torch.Tensor(word_del_freqs)
        if do_subsampling:
            self.word_freqs = self.word_freqs * (1. - self.word_del_freqs)

    def __len__(self):
        """
        返回整个数据集（所有单词）的长度
        """
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
        这个function返回以下数据用于训练
        - 中心词
        - 这个单词附近的(positive)单词
        - 随机采样的K个单词作为negative sample
        """
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - self.C, idx)) + list(range(idx + 1, idx + self.C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 怕超出 text 范围
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words
