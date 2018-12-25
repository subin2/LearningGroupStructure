
import datetime
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters

'''
    implementation of the co-cluster algorithm in the paper 'Minimum Sum-Squared Residue Co-clustering 
    of Gene Expression Data'
    H = (I − RRT)A(I − CCT)
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

def initialI(t):
    '''
    initialize the  matrix I used in co_cluster
    :param n: the row and column number
    '''
    # x, y = map(int, input().split())
    I = np.zeros((t, t), dtype=float)
    for i in range(0, t):
        I[i, i] = 1
    print(I)
    return np.matrix(I)


def H2(IR, IC, A, R, C):
    '''
        calculate the value || H ||^2, which is the sum-Squared residue of co-clustering result
    '''
    # H1 = R * R.T * A * C * C.T
    # H = A - H1
    HR = IR - R * R.T
    HC = IC - C * C.T
    H = HR * A * HC
    return np.sum(np.multiply(H, H))


def get_tau(A):
    '''
       adjustable parameter, the threshold of ending the co-clustering
    '''
    return np.sum(np.multiply(A, A)) * 0.001


def calAC(I, A, R, C):
    HR = I - R * R.T
    return HR * A * C


def calAPR(I, A, R, C):
    HR = I - R * R.T
    return HR * A

def calAR(I, A, R, C):
    HC = I - C * C.T
    return R.T * A * HC

def calAPC(I, A, R, C):
    HC = I - C * C.T
    return A * HC




def argmin_c(APR, AC, C, j):
    '''
        choose the nearest cluster for column j
    '''
    minH = 0
    min_c = 0
    for c in range(0, C.shape[1]):
        c_sum = np.sum(C[:, c])
        if c_sum != 0:
            c_sum = pow(c_sum, -1)
        H = APR[:, j] - c_sum * AC[:, c]
        HcJ = np.sum(np.multiply(H, H.T))
        if c == 0:
            minH = HcJ
        else:
            if HcJ < minH:
                minH = HcJ
                min_c = c
    return min_c


def argmin_r(APC, AR, R, i):
    '''
        choose the nearest cluster for row i
    '''
    minH = 0
    min_r = 0
    for r in range(0, R.shape[1]):
        r_sum = np.sum(R[:, r])
        if r_sum != 0:
            r_sum = pow(r_sum, -1)
        H = APC[i, :] - r_sum * AR[r, :]
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
    '''

    # record the start time
    start_time = datetime.datetime.now()
    # get rows and columns of  matrix A
    m, n = A.shape
    # initialize the cluster indicator matrix R and C, R for row cluster and C for column cluster.
    R = initialR_C(m, k)
    C = initialR_C(n, l)
    IR = initialI(m)
    IC = initialI(n)

    # calculate the sum-Squared residue
    objval = H2(IR, IC, A, R, C)
    # initial the threshold
    tau = get_tau(A)
    delta = tau + 1

    # do co-clustering
    while delta > tau:
        # clustering for columns
        AC = calAC(IR, A, R, C)
        APR = calAPR(IR, A, R, C)
        C_temp = np.zeros(C.shape)
        for j in range(0, n):
            c = argmin_c(APR=APR, AC=AC, C=C, j=j)
            C_temp[j, c] = 1

        # update cluster indicate matrix
        cluster_index_sum_C = np.sum(C_temp, axis=0)
        for s in range(0, l):
            if cluster_index_sum_C[s] != 0:
                cluster_index_sum_C[s] = pow(cluster_index_sum_C[s], -1 / 2)
        C = np.matrix(np.multiply(C_temp, cluster_index_sum_C))


        # clustering for rows
        AR = calAR(IC, A, R, C)
        APC = calAPC(IC, A, R, C)
        R_temp = np.zeros(R.shape)
        for i in range(0, m):
            r = argmin_r(APC=APC, AR=AR, R=R, i=i)
            R_temp[i, r] = 1
        # update cluster indicate matrix
        cluster_index_sum_R = np.sum(R_temp, axis=0)
        for v in range(0, k):
            if cluster_index_sum_R[v] != 0:
                cluster_index_sum_R[v] = pow(cluster_index_sum_R[v], -1 / 2)
        R = np.matrix(np.multiply(R_temp, cluster_index_sum_R))

        # update and calculate delta
        oldobj = objval
        objval = H2(IR, IC, A, R, C)
        delta = np.abs(oldobj - objval)
        print(delta)
    # when co-clustering ended, move the rows and columns according to the clustering result
    fit_A = A[np.argsort(np.sum(np.array(R.T), axis=0))]
    fit_A = fit_A[:, np.argsort(np.sum(np.array(C.T), axis=0))]
    plt.matshow(fit_A)

    # time cost
    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).seconds
    print("run cost time:")
    print(run_time)
    return np.matrix(fit_A)


# test
if __name__ == '__main__':
    data, rows, columns = make_biclusters(
        shape=(300, 300), n_clusters=3, noise=3,
        shuffle=True, random_state=0)
    plt.matshow(data)
    co_cluster(data, 3, 3)
    plt.show()
