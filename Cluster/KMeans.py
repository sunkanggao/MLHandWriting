# -*- coding:utf-8 -*-
import random
import math
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

def pearson(v1, v2):
    """
    计算两个向量之间的pearson相关系数。
    :param v1:
    :param v2:
    :return: 1 - pearson
    """
    u1 = sum(v1) / len(v1)
    u2 = sum(v2) / len(v2)
    cov = sum([(v1[i] - u1) * (v2[i] - u2) for i in range(len(v1))])
    std1 = math.sqrt(sum([pow(v1[i] - u1, 2) for i in range(len(v1))]))
    std2 = math.sqrt(sum([pow(v2[i] - u2, 2) for i in range(len(v2))]))
    return 1 - float(cov) / (std1 * std2)


def euclidean(v1, v2):
    """
    计算两向量之间的欧式距离。
    :param v1:
    :param v2:
    :return:
    """
    return  math.sqrt(sum([pow(v1[i] - v2[i], 2) for i in range(len(v1))]))


def k_means(rows, k, distance = pearson):
    # 确定每个特征的最小值和最大值
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
              for i in range(len(rows[0]))]

    # 随机创建k个中心
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(rows[0]))] for j in range(k)]

    bestmatches = None
    lastmatches = None
    for t in range(100):
        print 'Iteration %d' % (t+1)
        bestmatches = [[] for i in range(k)]

        # 在每一行中寻找距离最近的中心点
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)

        # 如果结果和上一次相同，则整个过程结束
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # 重置中心点
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    # return bestmatches
    clusters = [0 for i in range(len(rows))]
    for i in range(k):
        for j in range(len(bestmatches[i])):
            clusters[bestmatches[i][j]] = i + 1
    return clusters

if __name__ == '__main__':
    # np.random.seed(0)
    K, N = 3, 100
    data = np.empty((K * N, 2))
    means = [(-1, 2), (2, 2), (2, -2)]
    sigmas = [np.eye(2), 2 * np.eye(2), np.diag((1, 2))]
    for i in range(K):
        mn = stats.multivariate_normal(means[i], sigmas[i] * 0.2)
        data[i*N : (i+1)*N] = mn.rvs(N)
    clusters = k_means(data, 3, euclidean)
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap=mpl.colors.ListedColormap(['r', 'g', 'b']))
    plt.show()


