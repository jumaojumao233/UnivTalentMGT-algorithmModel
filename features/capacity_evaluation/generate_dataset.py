import pandas as pd
import numpy as np

# 生成模拟数据集
np.random.seed(42)

# 干部ID
staff_ids = [f"S{i:03}" for i in range(1, 101)]

# 教学特征
teaching_features = {
    'courseCount': np.random.randint(1, 10, 100),
    'studentEvaluation': np.random.uniform(3.0, 5.0, 100),
    'awards': np.random.randint(0, 5, 100)
}

# 科研特征
research_features = {
    'publications': np.random.randint(0, 20, 100),
    'patents': np.random.randint(0, 10, 100),
    'projectFunding': np.random.randint(0, 1000000, 100)
}

# 管理特征
management_features = {
    'position': np.random.choice(['Department Head', 'Team Leader', 'Manager'], 100),
    'teamSize': np.random.randint(5, 50, 100),
    'collaborations': np.random.randint(0, 10, 100)
}

# 创新特征
innovation_features = {
    'innovativeProjects': np.random.randint(0, 5, 100),
    'awards': np.random.randint(0, 3, 100)
}

# 目标评分
capabilities = {
    'teaching': np.random.uniform(3.0, 5.0, 100),
    'research': np.random.uniform(3.0, 5.0, 100),
    'management': np.random.uniform(3.0, 5.0, 100),
    'innovation': np.random.uniform(3.0, 5.0, 100)
}

# 创建数据框
data = pd.DataFrame({
    'staffId': staff_ids,
    **teaching_features,
    **research_features,
    **management_features,
    **innovation_features,
    **capabilities
})

# 保存数据集
data.to_csv('staff_capabilities.csv', index=False)