# -*- coding: utf-8 -*-
# @Time    : 18-12-24
# @Author  : Wuqiao Chen
# @Site    : https://github.com/Lokfar
# @File    : co_cluster_wuqiao.py
# @IDE     : PyCharm Community Edition

import datetime
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters


class BaseCoCluster:
    def __init__(self, w, threshold_factor=0.00001, row_cluster_num=2, column_cluster_num=2):
        '''
        init the co-cluster algorithm class
        :param w: the matrix to be co-clustered，height * width * in_channel * out_channel
        :param threshold_factor:
        :param row_cluster_num:
        :param column_cluster_num:
        '''
        if isinstance(w, np.ndarray) and len(w.shape) == 4:
            row_num, col_num, in_channel, out_channel = w.shape
            self.w = w
            self.row_num = row_num
            self.col_num = col_num
            self.in_channel = in_channel
            self.out_channel = out_channel
            self.threshold_facotor = threshold_factor
            self.k = row_cluster_num
            self.l = column_cluster_num
            self.run_time = 0
        else:
            raise Exception("wrong array")

    def initialR_C(self, p, q):
        '''
        initialize the cluster indicator matrix, Sequential allocation of the first k values and random allocation of the rest
        '''
        R = np.zeros((p, q), dtype=float)
        for i in range(0, p):
            if i < q:
                R[i, i] = 1
            else:
                random_c = random.randint(0, q - 1)
                R[i, random_c] = 1
        cluster_index_sum = np.sum(R, axis=0) ** (-1 / 2)
        return np.matrix(np.multiply(R, cluster_index_sum))

    def H2(self, A, R, C):
        '''
            calculate the value || H ||^2, which is the sum-Squared residue of co-clustering result
        '''
        H1 = R * R.T * A * C * C.T
        H = A - H1
        return np.sum(np.multiply(H, H))

    def get_tau(self, A):
        '''
           adjustable parameter, the threshold of ending the co-clustering
        '''
        return np.sum(np.multiply(A, A)) * self.threshold_facotor

    def calAC(self, A, R, C):
        return R * R.T * A * C

    def calAR(self, A, R, C):
        return R.T * A * C * C.T

    def argmin_c(self, A, AC, C, j):
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

    def argmin_r(self, A, AR, R, i):
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

    def col_mean(self, A, R, C):
        return R * R.T * A * (np.multiply(C, C))

    def row_mean(self, A, R, C):
        return (np.multiply(R.T, R.T)) * A * C * C.T

    def co_cluster_one(self, A):
        '''
        do cluster for a matrix ,for example 3 * 3
        do cluster for the matrix's row and column at the same time
        :param A: the matrix to be clustered
        :return:  the clustered matrix
        '''

        # get rows and columns of  matrix A
        # m, n = A.shape
        # initialize the cluster indicator matrix R and C, R for row cluster and C for column cluster.
        R = self.initialR_C(self.row_num, self.k)
        C = self.initialR_C(self.col_num, self.l)

        # calculate the sum-Squared residue
        objval = self.H2(A, R, C)
        # initial the threshold
        tau = self.get_tau(A)
        delta = tau + 1

        # do co-clustering
        while delta > tau:
            # clustering for columns
            AC = self.calAC(A, R, C)
            C_temp = np.zeros(C.shape)
            for j in range(0, self.col_num):
                c = self.argmin_c(A=A, AC=AC, C=C, j=j)
                C_temp[j, c] = 1

            # update cluster indicate matrix
            cluster_index_sum_C = np.sum(C_temp, axis=0)
            for s in range(0, self.l):
                if cluster_index_sum_C[s] != 0:
                    cluster_index_sum_C[s] = pow(cluster_index_sum_C[s], -1 / 2)
            C = np.matrix(np.multiply(C_temp, cluster_index_sum_C))

            # clustering for rows
            AR = self.calAR(A, R, C)
            R_temp = np.zeros(R.shape)
            for i in range(0, self.row_num):
                r = self.argmin_r(A=A, AR=AR, R=R, i=i)
                R_temp[i, r] = 1
            # update cluster indicate matrix
            cluster_index_sum_R = np.sum(R_temp, axis=0)
            for v in range(0, self.k):
                if cluster_index_sum_R[v] != 0:
                    cluster_index_sum_R[v] = pow(cluster_index_sum_R[v], -1 / 2)
            R = np.matrix(np.multiply(R_temp, cluster_index_sum_R))

            # update and calculate delta
            oldobj = objval
            objval = self.H2(A, R, C)
            delta = np.abs(oldobj - objval)

        row_means = self.row_mean(A, R, C)
        row_means_max_list = np.argmax(row_means, axis=0)
        row_mean_max_matrix = np.zeros((self.k, self.col_num), dtype=float)
        for i in range(0, self.col_num):
            row_mean_max_matrix[row_means_max_list[0, i], i] = 1
        R_one = np.where(R == 0, 0, 1)
        result = np.dot(R_one, row_mean_max_matrix)
        return result

    # get time cost
    def get_run_time(self):
        return self.run_time

    def co_cluster(self):
        # record the start time
        start_time = datetime.datetime.now()
        clustered_w = self.co_cluster_one(np.matrix(self.w[:, :, 0,0]))
        for i in range(0,self.out_channel):
            for j in range(0,self.in_channel):
                if i== 0 and j == 0:
                    continue
                clustered_w = np.concatenate((clustered_w, self.co_cluster_one(np.matrix(self.w[:, :, j, i]))), axis=0)
        result = clustered_w.reshape(self.out_channel,self.in_channel, self.row_num, self.col_num)
        # arrange the array to  height * width * in_channel * out_channel, for example 3 * 3* 128 * 128
        result = result.transpose((2, 3, 1, 0))
        # print(result)
        # time cost
        end_time = datetime.datetime.now()
        self.run_time = (end_time - start_time).seconds
        print("run cost time:")
        print(self.run_time)
        return result


if __name__ == '__main__':
    data_pre, rows_pre, columns_pre = make_biclusters(
        shape=(3, 4), n_clusters=2, noise=0,
        shuffle=True, random_state=0)
    print(data_pre)
    # data_pre = data_pre.reshape(3,3,1)
    for i in range(0, 6):
        for j in range(0,5):
            if i==0and j == 0:
                continue
            data, rows, columns = make_biclusters(
                shape=(3, 4), n_clusters=2, noise=0,
                shuffle=True, random_state=0)
            data_pre = np.concatenate((data_pre, data), axis=0)

    # print(data_pre)
    data_pre = data_pre.reshape(6, 5, 3, 4)
    # print(data_pre)
    data_pre = data_pre.transpose((2, 3, 1, 0))
    print(data_pre)
    print("onecheck")
    print(data_pre[:,:,0,0])
    baseCoCluster = BaseCoCluster(w=data_pre)
    result = BaseCoCluster.co_cluster(baseCoCluster)
    print("start")
    print(result)
    print("end")
