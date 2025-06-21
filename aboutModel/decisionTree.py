import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 加载数据
data = pd.read_csv(r"D:\西电第六学期\小学期\UnivTalentMGT-algorithmModel\aboutDataset\人员数据集.csv")
"""
姓名,岗位,能力标签,历史绩效,技能特长,培训记录,入职日期
"""

# 查看数据结构
print(data.head())

# 将分类变量转换为数值型
label_encoders = {}
# 提取入职日期中的年份和月份
data['入职年份'] = pd.to_datetime(data['入职日期']).dt.year
data['入职月份'] = pd.to_datetime(data['入职日期']).dt.month
# 假设目标变量是“是否适合岗位”，这里用历史绩效评分的中位数作为阈值来定义是否适合岗位
median_performance = data['历史绩效'].median()
data = data.drop(columns=['姓名', '入职日期'])
for column in ['岗位', '能力标签', '技能特长', '培训记录','入职年份','入职月份']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
joblib.dump(label_encoders,r"D:\西电第六学期\小学期\UnivTalentMGT-algorithmModel\aboutModel/saved_stander/decission_tree.pkl")
data['是否适合岗位'] = (data['历史绩效'] >= median_performance).astype(int)
data = data.drop(columns=['历史绩效'])

from models import getResultOfDecisionTree
print('example input:',dict(data.iloc[0,:-1]))
print(data.iloc[:,-1])
resultModel=getResultOfDecisionTree(data=data,plot=False)
print(resultModel)

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
#save model
model=resultModel['model']
joblib.dump(model, r"D:\西电第六学期\小学期\UnivTalentMGT-algorithmModel\aboutModel/saved_model/decission_tree.pkl")
# pickle.dump(scaler, open('scaler.pkl','wb'))
#load model
# rfc2 = joblib.load('saved_model/decission_tree.pkl')
# print(rfc2.predict(X[0:1,:]))
"""
决策树进行预测跑通了
预测的是{是否适合岗位}
"""