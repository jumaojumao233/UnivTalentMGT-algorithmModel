import pandas as pd
import numpy as np

# 生成模拟数据集
np.random.seed(42)

# 干部ID
staff_ids = [f"S{i:03}" for i in range(1, 101)]

# 基本信息
names = [f"姓名{i:03}" for i in range(1, 101)]
ages = np.random.randint(25, 60, 100)
genders = np.random.choice(['Male', 'Female'], 100)

# 职位
positions = np.random.choice(['Team Leader', 'Researcher', 'Developer', 'Manager'], 100)

# 技能
skills = [
    'Leadership', 'Project Management', 'Data Analysis', 'Machine Learning',
    'Programming', 'Communication', 'Problem Solving', 'Innovation'
]
staff_skills = [np.random.choice(skills, np.random.randint(1, 5)).tolist() for _ in range(100)]

# 工作经验
experience = np.random.randint(1, 20, 100)

# 教育水平
education_levels = np.random.choice(['Bachelor', 'Master', 'PhD'], 100)

# 创建数据框
data = pd.DataFrame({
    'staffId': staff_ids,
    'name': names,
    'age': ages,
    'gender': genders,
    'position': positions,
    'skills': staff_skills,
    'experience': experience,
    'educationLevel': education_levels
})

# 保存数据集
data.to_csv('staff_info.csv', index=False)