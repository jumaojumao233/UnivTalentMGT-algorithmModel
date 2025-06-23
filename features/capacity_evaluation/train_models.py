import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 加载数据集
data = pd.read_csv('staff_capabilities.csv')

# 特征和目标
#staffId,courseCount,studentEvaluation,awards,publications,patents,projectFunding,position,teamSize,collaborations,innovativeProjects,teaching,research,management,innovation
features = ['courseCount', 'studentEvaluation', 'awards', 'publications', 'patents', 'projectFunding',
            'teamSize', 'collaborations', 'innovativeProjects']
targets = ['teaching', 'research', 'management', 'innovation']

# 训练模型
models = {}
for target in targets:
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model for {target} - MSE: {mean_squared_error(y_test, y_pred)}")

    models[target] = model

# 保存模型
for target, model in models.items():
    joblib.dump(model, f'{target}_model.pkl')