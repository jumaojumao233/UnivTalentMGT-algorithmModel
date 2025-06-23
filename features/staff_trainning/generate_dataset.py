import pandas as pd
import numpy as np

# 生成模拟数据集
np.random.seed(42)

# 干部ID
staff_ids = [f"S{i:03}" for i in range(1, 101)]

# 基本信息
names = [f"姓名{i:03}" for i in range(1, 101)]

# 技能
skills = [
    'Data Analysis', 'Machine Learning', 'Programming', 'Communication',
    'Problem Solving', 'Innovation', 'Leadership', 'Project Management'
]
staff_skills = [np.random.choice(skills, np.random.randint(1, 5)).tolist() for _ in range(100)]

# 工作经验
experience = np.random.randint(1, 20, 100)

# 教育水平
education_levels = np.random.choice(['Bachelor', 'Master', 'PhD'], 100)

# 岗位要求
position_requirements = [
    {"requiredSkills": ["Data Analysis", "Machine Learning"], "minimumExperience": 3},
    {"requiredSkills": ["Leadership", "Project Management"], "minimumExperience": 5},
    {"requiredSkills": ["Programming", "Innovation"], "minimumExperience": 2}
]

# 创建数据框
data = pd.DataFrame({
    'staffId': staff_ids,
    'name': names,
    'skills': staff_skills,
    'experience': experience,
    'educationLevel': education_levels,
    'positionRequirements': [np.random.choice(position_requirements) for _ in range(100)]
})

# 保存数据集
data.to_csv('staff_training_info.csv', index=False)