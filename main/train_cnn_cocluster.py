# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : train.py
# @IDE     : PyCharm Community Edition
"""
    将em中的模型转化成co-cluster的综合版本, 并进行train测试
    todo: 1. 完善data_processor.py                     finished
    todo: 2. 完善cnn_cocluster_config.py, 设置全局参数   finished
    todo: 3. 建立一个前向传播模型 cnn_cocluster
    todo: 4. 使用co-cluster对W进行优化 cnn_cocluster
    todo: 5. 更新网络结构 cnn_cocluster
    todo: 6. 打印结果 ./tools/show_result.py
"""
from data_provider import data_processor

def train():
    # 获取数据集
    train_data, train_label, test_data, test_label = data_processor.get_data()




if __name__ == '__main__':
    train()