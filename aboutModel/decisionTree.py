import pandas as pd
import numpy as np
import decisionTree as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 加载数据
data = pd.read_csv("../aboutDataset/人员数据集.csv")

# 查看数据结构
print(data.head())

# 将分类变量转换为数值型
label_encoders = {}
for column in ['岗位', '能力标签', '技能特长', '培训记录']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 提取入职日期中的年份和月份
data['入职年份'] = pd.to_datetime(data['入职日期']).dt.year
data['入职月份'] = pd.to_datetime(data['入职日期']).dt.month

# 假设目标变量是“是否适合岗位”，这里用历史绩效评分的中位数作为阈值来定义是否适合岗位
median_performance = data['历史绩效'].median()
data['是否适合岗位'] = (data['历史绩效'] >= median_performance).astype(int)

# 删除不需要的列
data = data.drop(columns=['姓名', '入职日期', '历史绩效'])

from models import getResultOfDecisionTree
print(data.iloc[:,-1])
resultModel=getResultOfDecisionTree(data=data)
print(resultModel)
"""
决策树进行预测跑通了
预测的是{是否适合岗位}
"""