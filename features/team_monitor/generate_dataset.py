import pandas as pd
import numpy as np

# 生成模拟数据集
np.random.seed(42)

# 班子ID
team_ids = [f"T{i:03}" for i in range(1, 11)]

# 成员信息
current_staff = [
    {"staffId": f"S{i:03}", "position": np.random.choice(['Team Leader', 'Researcher', 'Developer', 'Manager'])}
    for i in range(1, 21)
]

# 近期变动
change_types = ['New Hire', 'Resignation', 'Promotion']
recent_changes = [
    {"staffId": f"S{i:03}", "changeType": np.random.choice(change_types), "position": np.random.choice(['Team Leader', 'Researcher', 'Developer', 'Manager'])}
    for i in range(21, 31)
]

# 创建数据框
data = pd.DataFrame({
    'teamId': np.random.choice(team_ids, 10),
    'currentStaff': [current_staff[:2] for _ in range(10)],
    'recentChanges': [recent_changes[:2] for _ in range(10)]
})

# 保存数据集
data.to_csv('team_monitoring_info.csv', index=False)