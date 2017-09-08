# -*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures


def softmax(data, K, alpha, lamda):
    """
    softmax回归，训练得到模型参数。
    :param data: 样本矩阵
    :param K: 类别个数
    :param alpha: 迭代步长
    :param lamda: L2正则化参数
    :return: 模型参数，K * feature
    """
    f = len(data[0]) - 1
    w = np.zeros((K, f))
    wNew = np.zeros((K, f))
    for times in range(100):
        for d in data:
            x = d[:-1]
            for k in range(K):
                y_k = (1 if d[-1] == k else 0)
                g = (y_k - calcp(w, x, k)) * x
                wNew[k] = w[k] + alpha * g + lamda * w[k]
            w = wNew.copy()
    return w


def calcp(w, x, k):
    """
    计算当前样本x在w参数下，属于k类的概率是多少。
    :param w: 参数矩阵
    :param x: 当前样本
    :param k: 类别序号
    :return: 样本x属于类别k的概率
    """
    K, j = w.shape
    a, b = np.exp(np.dot(w[k], x.T)), 0
    for l in range(K):
        b += np.exp(np.dot(w[l], x.T))
    return a / b


def predict(data, w, K):
    """
    给定样本集，预测对应样本类别，返回类别数组。
    :param data: 样本矩阵
    :param w: 参数矩阵
    :param K: 类别个数
    :return: 预测类别列表
    """
    y_hat = []
    for i in data:
        cur = []
        for j in range(K):
            cur.append(calcp(w, i, j))
        y_hat.append(cur.index(max(cur)))
    return np.array(y_hat)


def extend(a, b):
    big, small = 1.01, 0.01
    return big * a - small * b, big * b - small * a


if __name__ == '__main__':
    # 随机生成一个数据集，
    np.random.seed(0)
    K, N = 3, 100
    data = np.empty((K * N, 3))
    means = [(-1, 2), (2, 2), (2, -2)]
    sigmas = [np.eye(2), 2 * np.eye(2), np.diag((1, 2))]
    for i in range(K):
        mn = stats.multivariate_normal(means[i], sigmas[i] * 0.2)
        data[i*N : (i+1)*N, :-1] = mn.rvs(N)
        data[i*N : (i+1)*N, -1] = i
    ori = data

    # 对原训练样本进行多项式升维
    # poly = PolynomialFeatures(degree=2)
    # y = data[:, -1].reshape(-1, 1)
    # data = poly.fit_transform(data[:, :-1])
    # data = np.concatenate((data, y), axis=1)

    # 训练得到模型参数
    w = softmax(data, K, 0.1, 0)
    print u'模型参数为：\n', w

    # 对训练集计算误差
    y_hat = predict(data[:, :-1], w, K)
    y = data[:, -1]
    acc = np.mean(y == y_hat) * 100
    print u'准确率：%.2f%%' % (acc)

    # 可视化
    x1_min, x1_max = extend(ori[:, 0].min(), ori[:, 0].max())
    x2_min, x2_max = extend(ori[:, 1].min(), ori[:, 1].max())
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    # grid_test = poly.fit_transform(grid_test)
    test_y_hat = predict(grid_test, w, K)
    test_y_hat.shape = x1.shape

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.pcolormesh(x1, x2, test_y_hat, cmap=cm_light)
    plt.scatter(ori[:, 0], ori[:, 1], c = ori[:, -1], edgecolors='k', cmap=cm_dark)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title(u'softmax回归', fontsize=18)
    plt.tight_layout(0.2)
    plt.show()