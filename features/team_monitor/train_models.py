import pandas as pd
import numpy as np
import joblib

# 加载数据集
data = pd.read_csv('team_monitoring_info.csv')

# 将成员信息和近期变动转换为列表
data['currentStaff'] = data['currentStaff'].apply(lambda x: eval(x))
data['recentChanges'] = data['recentChanges'].apply(lambda x: eval(x))


# 风险预警与优化建议函数
def monitor_team(team_id, current_staff, recent_changes):
    # 找到对应的班子
    team = data[data['teamId'] == team_id].iloc[0]


    # 提取当前成员信息
    current_staff_positions = {member['position'] for member in current_staff}

    # 提取近期变动信息
    recent_changes_positions = {change['position'] for change in recent_changes}

    # 识别风险
    risk_warnings = []
    optimization_suggestions = []

    # 检查关键职位是否空缺
    key_positions = ['Team Leader', 'Project Manager']
    for position in key_positions:
        if position not in current_staff_positions:
            risk_warnings.append({
                "riskType": "Key Position Vacancy",
                "position": position,
                "severity": "High"
            })
            optimization_suggestions.append({
                "suggestion": f"Recruit a new {position} with at least 5 years of experience",
                "priority": "High"
            })

    # 检查近期变动
    for change in recent_changes:
        if change['changeType'] == 'Resignation':
            risk_warnings.append({
                "riskType": "Resignation",
                "position": change['position'],
                "severity": "Medium"
            })
            optimization_suggestions.append({
                "suggestion": f"Assign additional responsibilities to the current Team Leader",
                "priority": "Medium"
            })

    return risk_warnings, optimization_suggestions

def dumpModel():
    # 保存模型
    joblib.dump(monitor_team, 'monitor_team.pkl')

dumpModel()