import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 加载数据集
data = pd.read_csv('staff_training_info.csv')

# 将技能列表转换为字符串
data['skills'] = data['skills'].apply(lambda x: ' '.join(eval(x)))
data['positionRequirements'] = data['positionRequirements'].apply(lambda x: eval(x))

# 特征提取
vectorizer = CountVectorizer()
X_skills = vectorizer.fit_transform(data['skills'])


# 训练模型
def recommend_training(staff_id, capabilities, position_requirements):
    # 找到对应的干部
    staff = data[data['staffId'] == staff_id].iloc[0]

    # 提取干部技能
    staff_skills = set(staff['skills'].split())

    # 提取岗位要求
    required_skills = set(position_requirements['requiredSkills'])
    minimum_experience = position_requirements['minimumExperience']

    # 识别能力缺口
    training_needs = []
    for skill in required_skills:
        if skill not in staff_skills:
            training_needs.append(skill)

    # 生成培训计划
    training_plan = []
    for skill in training_needs:
        if skill == 'Data Analysis':
            training_plan.append({"courseName": "Advanced Data Analysis", "courseId": "DA101"})
        elif skill == 'Machine Learning':
            training_plan.append({"courseName": "Introduction to Machine Learning", "courseId": "ML101"})
        # 可以根据需要添加更多课程

    return training_needs, training_plan


# 保存模型
joblib.dump(recommend_training, 'recommend_training.pkl')