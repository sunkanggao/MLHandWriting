# -*- coding: utf-8 -*-

import numpy as np

def BGD(x, y, alpha, eps, maxInterations):
    """
    批量梯度下降
    :param x: 样本矩阵（m x n）
    :param y: 样本标签（m x 1）
    :param alpha: 迭代步长
    :param eps: 停止迭代的最小误差
    :param maxInterations: 最大迭代次数
    :return: 估计参数，迭代次数，误差
    """
    m, n = x.shape
    theta = np.ones((n, 1))
    loss = 10
    iter_count = 0
    while loss > eps and iter_count < maxInterations:
        h = np.dot(x, theta)
        lossCur = y - h
        theta = theta + alpha * np.dot(x.T, lossCur) / n
        iter_count += 1
        loss = (lossCur ** 2).sum() / m

    return theta, iter_count, loss



def SGD(x, y, alpha, eps, maxInterations):
    """
    随机梯度下降
    :param x: 样本矩阵（m x n）
    :param y: 样本标签（m x 1）
    :param alpha: 迭代步长
    :param eps: 停止迭代的最小误差
    :param maxInterations: 最大迭代次数
    :return: 估计参数，迭代次数，误差
    """
    m, n = x.shape
    theta = np.ones((n, 1))
    loss = 10
    iter_count = 0
    while loss > eps and iter_count < maxInterations:
        # 随机选择一个样本
        idx = np.random.randint(m)
        xCur = x[idx].reshape(1, -1)
        h = np.dot(xCur, theta)
        theta = theta + alpha * (y[idx] - h) * xCur.T
        iter_count += 1
        loss = ((y - np.dot(x, theta)) ** 2).sum() / m

    return theta, iter_count, loss



def MBGD(x, y, alpha, eps, maxInterations, batch):
    """
    小批量梯度下降
    :param x: 样本矩阵（m x n）
    :param y: 样本标签（m x 1）
    :param alpha: 迭代步长
    :param eps: 停止迭代的最小误差
    :param maxInterations: 最大迭代次数
    :param batch: 每个批量的样本数
    :return: 估计参数，迭代次数，误差
    """
    m, n = x.shape
    # 随机分成小批量，每批10个样本
    mask = np.random.shuffle(np.arange(m))
    x = x[mask].reshape(m / batch, batch, n)
    y = y[mask].reshape(m / batch, batch)
    theta = np.ones((n, 1))
    loss = 10
    iter_count = 0
    while loss > eps and iter_count < maxInterations:
        # 随机选择一个批量
        idx = np.random.randint(m / 10)
        xCur = x[idx]
        h = np.dot(xCur, theta)
        lossCur = y[idx].reshape(-1, 1) - h
        theta = theta + alpha * np.dot(xCur.T, lossCur) / batch
        iter_count += 1
        loss = (lossCur ** 2).sum() / batch

    return theta, iter_count, loss


if __name__ == '__main__':
    # np.random.seed(0)
    x = np.linspace(-20, 20, 1000)
    np.random.shuffle(x)
    x = x.reshape(200, 5)
    para = np.array([[2, -1, 1.3, 4.2, -0.5]])
    y = np.dot(x, para.T)
    # 增加样本扰动
    # x = x + np.random.randn(200, 5) / 10
    print u'目标系数为', para.reshape((1, -1))

    theta, iter_count, loss = BGD(x, y, 0.0001, 10e-5, 1000)
    print u'\n**********批量梯度下降**********'
    print u'theta =', theta.reshape((1, -1))
    print u'迭代次数 =', iter_count
    print u'误差 =', loss

    theta, iter_count, loss = SGD(x, y, 0.001, 10e-5, 1000)
    print u'\n**********随机梯度下降**********'
    print u'theta =', theta.reshape((1, -1))
    print u'迭代次数 =', iter_count
    print u'误差 =', loss

    theta, iter_count, loss = MBGD(x, y, 0.001, 10e-5, 1000, 10)
    print u'\n**********小批量梯度下降**********'
    print u'theta =', theta.reshape((1, -1))
    print u'迭代次数 =', iter_count
    print u'误差 =', loss


"""
结果：
目标系数为 [[ 2.  -1.   1.3  4.2 -0.5]]

**********批量梯度下降**********
theta = [[ 1.99978397 -0.99976564  1.30006578  4.1997647  -0.50000153]]
迭代次数 = 14
误差 = 6.95925532088e-05

**********随机梯度下降**********
theta = [[ 1.99942506 -1.00024748  1.29973728  4.19960353 -0.50025178]]
迭代次数 = 75
误差 = 8.88611418639e-05

**********小批量梯度下降**********
theta = [[ 1.99926738 -0.99957682  1.30000006  4.19930271 -0.50026223]]
迭代次数 = 59
误差 = 8.75075190648e-05
"""