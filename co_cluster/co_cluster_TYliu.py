''' 
尝试将local search step与第一步算法结合
'''

import datetime
import random
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters

'''
    implementation of the co-cluster algorithm in the paper 'Minimum Sum-Squared Residue Co-clustering 
    of Gene Expression Data'
    H = A - RR'ACC'
    the adjustable parameter is tau
'''


def initialR_C(m, k):
    '''
    initialize the cluster indicator matrix, Sequential allocation of the first k values and random allocation of the rest
    :param m: the row number
    :param k: the column number
    '''
    R = np.zeros((m, k), dtype=float)
    for i in range(0, m):
        if i < k:
            R[i, i] = 1
        else:
            random_c = random.randint(0, k - 1)
            R[i, random_c] = 1
    cluster_index_sum = np.sum(R, axis=0) ** (-1 / 2)
    return np.matrix(np.multiply(R, cluster_index_sum))


def H2(A, R, C):
    '''
        calculate the value || H ||^2, which is the sum-Squared residue of co-clustering result
    '''
    H1 = R * R.T * A * C * C.T
    H = A - H1
    return np.sum(np.multiply(H, H))


def get_tau(A):
    '''
       adjustable parameter, the threshold of ending the co-clustering
    '''
    return np.sum(np.multiply(A, A)) * 0.00000001


def calAC(A, R, C):
    return R * R.T * A * C


def calAR(A, R, C):
    return R.T * A * C * C.T


def argmin_c(A, AC, C, j):
    '''
        choose the nearest cluster for column j
    '''
    minH = 0
    min_c = 0
    for c in range(0, C.shape[1]):
        c_sum = np.sum(C[:, c])
        if c_sum != 0:
            c_sum = pow(c_sum, -1)
        H = A[:, j] - c_sum * AC[:, c]
        HcJ = np.sum(np.multiply(H, H.T))
        if c == 0:
            minH = HcJ
        else:
            if HcJ < minH:
                minH = HcJ
                min_c = c
    return min_c


def argmin_r(A, AR, R, i):
    '''
        choose the nearest cluster for row i
    '''
    minH = 0
    min_r = 0
    for r in range(0, R.shape[1]):
        r_sum = np.sum(R[:, r])
        if r_sum != 0:
            r_sum = pow(r_sum, -1)
        H = A[i, :] - r_sum * AR[r, :]
        HIr = np.sum(np.multiply(H, H))
        if r == 0:
            minH = HIr
        else:
            if HIr < minH:
                minH = HIr
                min_r = r
    return min_r


def co_cluster(A, k, l):
    '''
    do cluster for the matrix's row and column at the same time
    :param A: the matrix to be clustered
    :param k: the number of row clusters
    :param l: the number of column clusters
    :return:  the clustered matrix
    '''

    # record the start time
    start_time = datetime.datetime.now()
    # get rows and columns of  matrix A
    m, n = A.shape
    # initialize the cluster indicator matrix R and C, R for row cluster and C for column cluster.
    R = initialR_C(m, k)
    C = initialR_C(n, l)

    # calculate the sum-Squared residue
    objval = H2(A, R, C)
    # initial the threshold
    tau = get_tau(A)
    delta = tau + 1

    # do co-clustering
    while delta > tau:
        # clustering for columns
        AC = calAC(A, R, C)
        C_temp = np.zeros(C.shape)
        for j in range(0, n):
            c = argmin_c(A=A, AC=AC, C=C, j=j)
            C_temp[j, c] = 1

        # update cluster indicate matrix
        cluster_index_sum_C = np.sum(C_temp, axis=0)
        for s in range(0, l):
            if cluster_index_sum_C[s] != 0:
                cluster_index_sum_C[s] = pow(cluster_index_sum_C[s], -1 / 2)
        C = np.matrix(np.multiply(C_temp, cluster_index_sum_C))

        # clustering for rows
        AR = calAR(A, R, C)
        R_temp = np.zeros(R.shape)
        for i in range(0, m):
            r = argmin_r(A=A, AR=AR, R=R, i=i)
            R_temp[i, r] = 1
        # update cluster indicate matrix
        cluster_index_sum_R = np.sum(R_temp, axis=0)
        for v in range(0, k):
            if cluster_index_sum_R[v] != 0:
                cluster_index_sum_R[v] = pow(cluster_index_sum_R[v], -1 / 2)
        R = np.matrix(np.multiply(R_temp, cluster_index_sum_R))

        # update and calculate delta
        oldobj = objval
        objval = H2(A, R, C)
        delta = np.abs(oldobj - objval)

    # when co-clustering ended, move the rows and columns according to the clustering result
    fit_A = A[np.argsort(np.sum(np.array(R.T), axis=0))]
    fit_A = fit_A[:, np.argsort(np.sum(np.array(C.T), axis=0))]

    # time cost
    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).seconds
    print("run cost time:")
    print(run_time)
    return np.matrix(local_search(fit_A, R.getA(), C.getA())), np.matrix(fit_A)


def r(row, matrix_c):
    for j in range(0, len(matrix_c[0])):
        if matrix_c[row, j] != 0:
            return j


def local_search(matrix_a, matrix_r, matrix_c):
    '''
    local_search step for column
    matrix_r为行聚类矩阵
    matrix_c为列聚类矩阵
    matrix_a为原矩阵
    r 为聚类映射函数
    '''
    if not isinstance(matrix_a, np.ndarray):
        print("matrix_a is not a ndarray")
        return
    if not isinstance(matrix_r, np.ndarray):
        print("matrix_r is not a ndarray")
        return
    if not isinstance(matrix_c, np.ndarray):
        print("matrix_c is not a ndarray")
        return

    # n l 分别为列聚类矩阵的行列
    n = len(matrix_c)
    l = len(matrix_c[0])

    # 阈值确定
    # 一范数
    matrix_a_norm = np.linalg.norm(matrix_a, ord=1, axis=None, keepdims=False)
    t = math.pow(10, -5) * math.pow(matrix_a_norm, 2)

    # A heat
    matrix_a_h = np.dot(matrix_r.T, matrix_a)

    list = []

    for i in range(0, n - 1):
        for c_ in range(0, l - 1):
            # 如果第i行的聚类结果不是c_类，则将其改变到c_类并计算损失函数
            if r(i, matrix_c) != c_:
                # 改变列聚类矩阵，将i行移入第c_类
                new_matrix_c = move_cluster(i, c_, matrix_c)

                # 4.13 左部
                cost = np.linalg.norm(np.dot(matrix_a_h, new_matrix_c)) - np.linalg.norm(np.dot(matrix_a_h, matrix_c))

                # 将行、列、损失函数值作为元组放入列表
                record = (i, c_, cost)
                list.append(record)

    # 获取列表的第三个元素
    def take_third(elem):
        return elem[2]

    # 将记录按照损失函数降序排列
    list.sort(key=take_third, reverse=True)
    best_movement = list[0]
    ret_matrix_c = matrix_c

    # 如果损失函数大于阈值，则按照该record移动，体现为修改matrix_c
    if best_movement[2] > t:
        ret_matrix_c = move_cluster(best_movement[0], best_movement[1], matrix_c)

    return ret_matrix_c


def move_cluster(sor_row, dest_col, ori_matrix_c):
    new_matrix_c = ori_matrix_c
    # 计算分类矩阵中目标列和来源行的mr值
    # 如果出现了空簇就设置为1
    m = 1
    for i in range(0, len(new_matrix_c) - 1):
        if new_matrix_c[i][dest_col] != 0:
            m = new_matrix_c[i][dest_col]
            break
    m_r_dest = math.pow(float(m), -2)

    sor_col = 0
    for i in range(0, len(new_matrix_c[0]) - 1):
        m = 0
        if new_matrix_c[sor_row][i] != 0:
            m = new_matrix_c[sor_row][i]
            sor_col = i
            break
    m_r_sor = math.pow(m, -2)

    # 对应数目变化
    m_r_dest = m_r_dest + 1
    m_r_sor = m_r_sor - 1

    # 更新分布矩阵
    for i in range(0, len(new_matrix_c) - 1):
        if new_matrix_c[i][dest_col] != 0:
            new_matrix_c[i][dest_col] = math.pow(m_r_dest, -0.5)
        if new_matrix_c[i][sor_col] != 0:
            new_matrix_c[i][sor_col] = math.pow(m_r_sor, -0.5)

    return new_matrix_c


# test
if __name__ == '__main__':
    data, rows, columns = make_biclusters(
        shape=(500, 500), n_clusters=10, noise=0,
        shuffle=True, random_state=0)
    plt.matshow(data)
    fit_A_1, fit_A = co_cluster(data, 10, 10)
    plt.matshow(fit_A)
    plt.matshow(fit_A_1)
    plt.show()
