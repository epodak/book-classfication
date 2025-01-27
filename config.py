import torch
import os
import numpy as np
import logging
from logging import handlers
import time
from datetime import timedelta

curPath = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件路径
root_path = os.path.abspath(os.path.dirname(__file__))  # 获取根目录路径
print('curPath:' + curPath)
print('root_path:' + root_path)

train_file = root_path + '/data/train_clean.csv'  # 训练数据文件路径
dev_file = root_path + '/data/dev_clean.csv'  # 验证数据文件路径
test_file = root_path + '/data/test_clean.csv'  # 测试数据文件路径
stopWords_file = root_path + '/data/stopwords.txt'  # 停用词文件路径
log_dir = root_path + '/logs/'  # 日志目录路径

bert_path = root_path + '/bert/'  # BERT模型路径
save_path = root_path + '/model/bert.ckpt'  # 保存模型路径
is_cuda = False  # 是否使用GPU
device = torch.device('cuda') if is_cuda else torch.device('cpu')  # 设置设备

class_list = [
    x.strip() for x in open(root_path + '/data/class.txt', encoding='utf-8').readlines()
]  # 类别名单
print('class_list: {}'.format(class_list))

# generate dl config
embedding = 'random'  # 词嵌入方式
embedding_pretrained = torch.tensor(
    np.load(root_path + '/data/' + embedding)["embeddings"].astype('float32')) \
    if embedding != 'random' else None  # 预训练的词嵌入矩阵

num_classes = len(class_list)  # 类别数量

num_epochs = 30  # epoch数
batch_size = 1  # mini-batch大小
pad_size = 400  # 每句话处理成的长度(短填长切)
learning_rate = 2e-5  # 学习率
dropout = 1.0  # 随机失活
require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
n_vocab = 50000  # 词表大小，在运行时赋值
embed = 300  # 向量维度
hidden_size = 768  # lstm隐藏层
num_layers = 1  # lstm层数
eps = 1e-8
max_length = 400
dim_model = 300
hidden = 1024
last_hidden = 512
num_head = 5
num_encoder = 2

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)  # 创建logger对象
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 设置日志格式
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)  # 把对象加到logger里

    return logger