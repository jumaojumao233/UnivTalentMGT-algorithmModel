import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 加载数据集
data = pd.read_csv('staff_info.csv')

# 将技能列表转换为字符串
data['skills'] = data['skills'].apply(lambda x: ' '.join(eval(x)))

# 特征提取
vectorizer = CountVectorizer()
X_skills = vectorizer.fit_transform(data['skills'])

# 训练模型
def recommend_staff(team_requirements):
    recommended_staff = []
    for position in team_requirements['positions']:
        position_name = position['name']
        required_skills = ' '.join(position['skills'])
        required_experience = position.get('experience', 0)
        required_education = position.get('educationLevel', 'Bachelor')

        # 计算技能匹配度
        required_skills_vector = vectorizer.transform([required_skills])
        skill_similarity = cosine_similarity(required_skills_vector, X_skills).flatten()

        # 筛选符合条件的干部
        filtered_data = data[
            (data['experience'] >= required_experience) &
            (data['educationLevel'] == required_education)
        ]

        # 计算匹配分数
        filtered_data['matchScore'] = skill_similarity[filtered_data.index]
        filtered_data = filtered_data.sort_values(by='matchScore', ascending=False)

        # 选择匹配度最高的干部
        if not filtered_data.empty:
            recommended_staff.append({
                'staffId': filtered_data.iloc[0]['staffId'],
                'name': filtered_data.iloc[0]['name'],
                'position': position_name,
                'matchScore': filtered_data.iloc[0]['matchScore']
            })

    return recommended_staff

# 保存模型
joblib.dump(recommend_staff, 'recommend_staff.pkl')