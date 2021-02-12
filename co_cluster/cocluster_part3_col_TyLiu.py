# -*-coding:utf-8-*
import numpy as np
import math

# local_search step for column
# matrix_r为行聚类矩阵
# matrix_c为列聚类矩阵
# matrix_a为原矩阵
# r 为聚类映射函数
def local_search(matrix_r, matrix_a, matrix_c, r):
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
    t = 10 ^ -5 * matrix_a_norm ^ 2

    # A heat
    matrix_a_h = np.dot(matrix_r.T, matrix_a)

    list = []

    for i in range(0, n - 1):
        for c_ in range(0, l - 1):
            # 如果第i行的聚类结果不是c_类，则将其改变到c_类并计算损失函数
            if r(i) != c_:
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
    m = 0
    for i in range(0, len(new_matrix_c) - 1):
        if new_matrix_c[i][dest_col] != 0:
            m = new_matrix_c[i][dest_col]
            break
    m_r_dest = math.pow(m, -2)

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
        if new_matrix_c[dest_col][i] != 0:
            new_matrix_c[dest_col][i] = math.pow(m_r_dest, -0.5)
        if new_matrix_c[sor_col][i] != 0:
            new_matrix_c[sor_col][i] = math.pow(m_r_sor, -0.5)

    return new_matrix_c
