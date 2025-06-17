from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
import decisionTree as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


class Models:
    def __init__(self):
        pass

    def getXGBoost(self):
        return xgb.XGBRegressor(
            max_depth=3,
            eta=0.3,
            objective='reg:squarederror',
            nthread=4,
            random_state=42
        )
    def getXGBoostRF(self):
        return xgb.XGBRFRegressor(random_state=42)

    def getCatBoost(self):
        return CatBoostRegressor(
            max_depth=3,
            learning_rate=0.3,
            loss_function='RMSE',
            thread_count=4,
            random_seed=42
        )

    def getGradientBoosting(self):
        return GradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.3,
            loss='absolute_error',
            n_estimators=100,
            random_state=42
        )

    def getLightGBM(self):
        model = lgb.LGBMRegressor(
            boosting_type='gbdt',  # 使用传统的决策树提升方法
            objective='regression',  # 回归任务
            metric='rmse',  # 评估指标为均方根误差
            num_leaves=31,  # 叶子节点数
            learning_rate=0.05,  # 学习率
            n_estimators=100  # 迭代次数
        )
        return model

    def getExtraTrees(self):
        return ExtraTreesRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )

    def getBagging(self):
        return BaggingRegressor(
            base_estimator=None,
            n_estimators=100,
            max_samples=1.0,
            random_state=42
        )

    def getNeuralNetwork(self):
        return MLPRegressor(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            random_state=42
        )
    def getDecisionTree(self):
        return DecisionTreeRegressor(random_state=0)
def getResultOfDecisionTree(data):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    import numpy as np
    import shap
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt

    modelResult={}

    X = data.iloc[:, :-1]  # 所有行，除最后一列的所有列作为特征

    y = data.iloc[:, -1]  # 最后一列作为目标变量

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = Models()
    # model=models.getXGBoost()
    model=models.getDecisionTree()

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    preds = model.predict(X_test)

    # 计算 MSE 和 RMSE
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae=mean_absolute_error(y_test,preds)
    r2=r2_score(y_test,preds)
    print(f'XGBoost MAE: {mae}')
    print(f'XGBoost MSE: {mse}')
    print(f'XGBoost RMSE: {rmse}')
    print(f'XGBoost R2: {r2}')

    # 获取特征重要性
    feature_importances = model.feature_importances_
    print(f'Feature importances: {feature_importances}')

    # 特征名称
    featureNamesII=data.columns.values[:-1]
    # 绘制柱状图
    plt.barh(range(len(feature_importances)), feature_importances, color='skyblue')
    plt.yticks(range(len(feature_importances)), featureNamesII)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis()  # 反转 y 轴，使得最重要的特征在上方
    plt.show()

    ##
    # 按值从大到小排序
    dffeimp=pd.DataFrame({
        'featureNames':featureNamesII,
        'feature_importances':feature_importances
    })
    df = dffeimp.sort_values(by='feature_importances', ascending=False)

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.barh(df['featureNames'], df['feature_importances'], color='skyblue')
    plt.xlabel('featureNames')
    plt.ylabel('feature_importances')
    plt.title('feature importances plot')
    plt.gca().invert_yaxis()
    plt.show()

    ##
    # SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # 排列图permutation plot
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx], color='b')
    plt.xlabel("Permutation Importance")
    plt.show()


    # 部分依赖图
    from matplotlib import pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    featureNames=data.columns.values[:-1]

    # Create and plot the data
    disp1 = PartialDependenceDisplay.from_estimator(model, X_train, featureNames)
    plt.subplots_adjust(wspace=0.4, hspace=1)
    plt.show()

    #预测值真实值比对图
    plt.scatter(y_test,preds)
    # 绘制参考线
    min_val = min(min(preds), min(y_test))
    max_val = max(max(preds), max(y_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # 黑色虚线
    # plt.xlabel("Feature Values")
    plt.xlabel("actual values")
    plt.ylabel("predations values")
    plt.show()

    predDataFrame=X_test
    predDataFrame.insert(loc=X_test.shape[1], column=data.columns.values[-1], value=preds)
    modelResult['result']=predDataFrame
    modelResult['mae']=mae
    modelResult['mse']=mse
    modelResult['rmse']=rmse
    modelResult['r2']=r2
    modelResult['feature_importances']=feature_importances
    # modelResult['shap_values']=shap_values
    modelResult['permutation_importance']=result
    return modelResult