# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 打开.log文件进行读取
    with open('./logs/word-embedding.log', 'r') as file:
        lines = file.readlines()

    # 初始化空列表来存储提取的数据
    epochs = []
    iters = []
    losses = []

    # 遍历每一行并提取数据
    for line in lines:
        parts = line.split(', ')
        for part in parts:
            key, value = part.split(': ')
            if key == 'epoch':
                epochs.append(int(value))
            elif key == 'iter':
                iters.append(int(value))
            elif key == 'loss':
                losses.append(float(value))
    print(len(losses))

    losses_wo_subsampling = losses[: len(losses) // 2]  # 前半部分没有 subsampling 的
    losses_w_subsampling = losses[len(losses) // 2:]    # 后半部分是有 subsampling 的
    epochs = list(range(0, 75))

    plt.plot(epochs, losses_w_subsampling, marker='o', linestyle='-', color='b', label='loss')
    plt.plot(epochs, losses_wo_subsampling, marker='o', linestyle='-', color='r', label='loss_wo_subsampling')


    # 添加标题和轴标签
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()
