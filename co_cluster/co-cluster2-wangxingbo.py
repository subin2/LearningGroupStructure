# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 12:29
# @Author  : WangXingbo
# @Site    : 
# @File    : co-cluster2-wangxingbo.py
# @Software: PyCharm
import datetime
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_biclusters

'''''
        supposing H =(I-RR')A(I-CC') from the paper Minimum Sum-Squared Residue Co-clustering 
    of Gene Expression Data
'''''


def co_cluster(A, k, l):
    """
    :param A:  data matrix A
    :param k:   row k
    :param l:   column l
    :return:
    """

    start_time = datetime.datetime.now()
    m, n = A.shape
    R = initialize(m, k)
    C = initialize(n, l)

    objval = H2(A, R, C)

    tau = get_tau(A)
    delta = 1
    while delta > tau:
        AC = calAC(A, R, C)
        AP1 = calAP1(A, R)
        C_temp = np.zeros(C.shape)
        for j in range(n):
            min_c = argmin_C(j, AP1, C, AC)
            C_temp[j,min_c] =1
        element_C = np.sum(C_temp, axis=0) ** (-1/2)
        for jc in range(C_temp.shape[1]):
            C_temp[:,jc] = C_temp[:, jc] * element_C[jc]
        # update C
        C = C_temp

        AR = calAR(A, R, C)
        AP2 = calAP2(A, R, C)
        R_temp  = np.zeros(R.shape)
        for i  in range(m) :
            min_r = argmin_R(i, AP2, R, AR)
            R_temp [i , min_r] =1
        element_R = np.sum(R_temp, axis=0) **(-1/2)
        for ir in range (R_temp.shape[1]):
            R_temp[:, ir] = R_temp[:, ir] * element_R[ir]
        #update R
        R = R_temp

        oldobj = objval
        objval = H2(A, R , C)
        delta =np.abs(oldobj - objval)
    end_time = datetime.datetime.now()
    cost = (end_time - start_time).seconds




def initialize(m, n):
    """
    :param m: row
    :param n: column
    :return:  initialized matrix R or C (R and C are the indicator matrix in the paper)
    """
    initR = np.zeros((m, n), dtype=float)
    for i in range(m):
        if i < n:
            initR[i, i] = 1
        else:
            random = random.randint(0, n - 1)
            R[i, random] = 1
    element = np.sum(R, axis=0) ** (-1 / 2)# column r  of R has mr non-zeros whic equal mr(-1/2);  and  for m1 +  ... mk = m;

    for j in range(n):
            initR[:,j] = initR[:,j] * element[j]

    return  initR



def get_tau(A):
    '''
       adjustable parameter
    '''
    return np.sum(np.multiply(A, A)) * 0.0000000001

def H2(A, R, C):
    """
    :type C: object
    :return: objective function ||H||2
    """
    i_r = np.identity(A.shape[0])  # number of the rows
    i_c = np.identity(A.shape[1])  # number of the columns
    h = (i_r-R * R.T) * A * (i_c - C * C.T)
    return np.sum(np.multiply(h, h))

def calAC(A, R, C):
    """
     :return:  before update C
     """
    i_r = np.identity(A.shape[0])
    return (i_r - R * R.T) * A * C


def calAP1(A, R):
    """
    :return:  before update C
    """
    i_r = np.identity(A.shape[0])
    return (i_r - R * R.T) * A


def calAP2(A, R, C):
    """
    :return: before  update R
    """
    i_c = np.identity(A.shape[1])
    return A * (i_c - C * C.T)

def calAR(A, R, C):
    """
    :return: before  update R
    """
    i_c = np.identity(A.shape[1])
    return R.T * A * (i_c - C * C.T)

def argmin_C(j, AP1, C, AC):
    """
    :return:  classify the column j using armin||AP.j - pow(nc, -1/2)* AC.c ||2 in the paper
    """


    old_sum_obj_f =0
    min_c = 0
    for c in range(C.shape[1]):# 0~ l-1
        norm1 = np.linalg.norm(C[:,c],1) #  某一列第一范数的平方为nr
        nc = pow(norm1, 2)# nc
        obj_f = AP1[:,j] -  pow(nc, -1/2) * AC [:, c]
        sum_obj_f = np.sum(np.multiply(obj_f, obj_f.T))

        if c == 0:
            old_sum_obj_f = sum_obj_f
        else:
            if old_sum_obj_f> sum_obj_f:
                old_sum_obj_f = sum_obj_f
                min_c = c

    return  min_c





def argmin_R(i, AP2, R, AR):
    """
    :return: classify the row i using armin||AP2i. - pow(mr, -1/2)* ARr. ||2 in the paper
    """
    old_sum_obj_f = 0
    min_r = 0
    for r in range(R.shape[1]):  # 0~ n-1
        norm1 = np.linalg.norm(R[:,r], 1)  # 某一列第一范数的平方为mr
        mr = pow(norm1, 2)  # mr
        obj_f = AP2[i,:] - pow(mr, -1 / 2) * AR[r,:]
        sum_obj_f = np.sum(np.multiply(obj_f, obj_f.T))

        if c == 0:
            old_sum_obj_f = sum_obj_f
        else:
            if old_sum_obj_f > sum_obj_f:
                old_sum_obj_f = sum_obj_f
                min_r = c

    return min_r

