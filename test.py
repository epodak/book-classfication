from predict import Predict  # 导入Predict类
from config import root_path  # 导入root_path变量
import json  # 导入json模块

if __name__ == '__main__':  # 主程序运行条件
    text = '装帧全部采用日本进口竹尾纸，专为读书人打造奢华手感 ◆ 畅销100万册，独占同名书市场七成份额...'
    predict = Predict()  # 创建Predict类的实例对象
    label, score = predict.predict(text)  # 调用predict方法，获取预测结果的标签和分数
    print('label:{}'.format(label))  # 打印预测结果的标签
    print('score:{}'.format(score))  # 打印预测结果的分数
    with open(root_path + '/data/label2id.json', 'r') as f:  # 打开文件"label2id.json"，保存在变量f中
        label2id = json.load(f)  # 读取文件内容，并保存在label2id变量中
    print(list(label2id.keys())[list(label2id.values()).index(label)])  # 打印label在label2id中的对应键，即label的名称