# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : train.py
# @IDE     : PyCharm Community Edition
"""
    将em中的模型转化成co-cluster的综合版本, 并进行train测试
    todo: 1. 完善data_processor.py                     finished
    todo: 2. 完善cnn_cocluster_config.py, 设置全局参数   finished
    todo: 3. 搭建神经网络
"""
from data_provider import data_processor

def train():
    # 获取数据集
    train_data, train_label, test_data, test_label = data_processor.get_data()




if __name__ == '__main__':
    train()