import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 加载数据集
data = pd.read_csv('staff_tags.csv')

# 特征和目标
features = ['teaching', 'research', 'management', 'innovation', 'yearsOfExperience']
capability_target = 'capability_tags'
potential_target = 'potential_tags'
development_target = 'development_tags'

# 数据标准化
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 训练能力标签模型
X_capability = data[features]
y_capability = data[capability_target]

capability_model = LogisticRegression(max_iter=1000)
capability_model.fit(X_capability, y_capability)

# 训练潜力标签模型
X_potential = data[features]
y_potential = data[potential_target]

potential_model = LogisticRegression(max_iter=1000)
potential_model.fit(X_potential, y_potential)

# 训练发展标签模型
X_development = data[features]
y_development = data[development_target]

development_model = LogisticRegression(max_iter=1000)
development_model.fit(X_development, y_development)

# 保存模型
joblib.dump(capability_model, 'capability_model.pkl')
joblib.dump(potential_model, 'potential_model.pkl')
joblib.dump(development_model, 'development_model.pkl')