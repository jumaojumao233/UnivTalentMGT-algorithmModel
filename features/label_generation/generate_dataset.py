import pandas as pd
import numpy as np

# 生成模拟数据集
np.random.seed(42)

# 干部ID
staff_ids = [f"S{i:03}" for i in range(1, 101)]

# 能力评分
capabilities = {
    'teaching': np.random.uniform(3.0, 5.0, 100),
    'research': np.random.uniform(3.0, 5.0, 100),
    'management': np.random.uniform(3.0, 5.0, 100),
    'innovation': np.random.uniform(3.0, 5.0, 100)
}

# 其他信息
additional_info = {
    'yearsOfExperience': np.random.randint(1, 20, 100),
    'educationLevel': np.random.choice(['Bachelor', 'Master', 'PhD'], 100)
}

# 标签
tags = {
    'capability_tags': np.random.choice(['教学能手', '科研骨干', '管理专家'], 100),
    'potential_tags': np.random.choice(['高潜后备', '创新先锋'], 100),
    'development_tags': np.random.choice(['需强化科研', '管理能力待提升', '创新转化不足'], 100)
}

# 创建数据框
data = pd.DataFrame({
    'staffId': staff_ids,
    **capabilities,
    **additional_info,
    **tags
})

# 保存数据集
data.to_csv('staff_tags.csv', index=False)