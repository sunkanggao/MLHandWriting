# -*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def softmax(data, K, alpha, lamda):
    n = len(data[0]) - 1
    w = np.zeros((K, n))
    wNew = np.zeros((K, n))
    for times in range(1000):
        for d in data:
            x = d[:-1]
            for k in range(K):
                y = 0
                if int(d[-1]) == k:
                    y = 1
                p = predict(w, x, k)
                g = (y - p) * x
                wNew[k] = w[k] + alpha * g + lamda * w[k]
            w = wNew.copy()
    return w


def predict(w, x, k):
    K, j = w.shape
    a = np.exp(w[k] * x.T)
    b = 0
    for l in range(K):
        b += np.exp(w[l] * x.T)
    return a / b


if __name__ == '__main__':
    # 随机生成一个数据集，
    np.random.seed(0)
    K = 3
    N = 50
    data = np.empty((K * N, 3))
    means = [(-1, 2), (2, 2), (2, -2)]
    sigmas = [np.eye(2), 2 * np.eye(2), np.diag((1, 2))]
    for i in range(K):
        mn = stats.multivariate_normal(means[i], sigmas[i] * 0.3)
        data[i*N : (i+1)*N, :-1] = mn.rvs(N)
        data[i*N : (i+1)*N, -1] = i
    w = softmax(data, K, 0.2, 0.1)
    print w
    plt.scatter(data[:, 0], data[:, 1], c = data[:, -1], cmap=plt.cm.Set1)
    plt.show()