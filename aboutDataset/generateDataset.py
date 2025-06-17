import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# 生成随机日期
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))


# 定义可能的岗位和技能
positions = ["教学岗位", "科研岗位", "管理岗位"]
skills = ["沟通能力", "团队协作", "数据分析", "项目管理", "教学设计", "科研能力", "领导力"]
training_records = ["课程A", "课程B", "课程C", "课程D", "无"]


# 生成模拟数据
def generate_dataset(num_records=100):
    data = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()

    for i in range(num_records):
        name = f"员工{i + 1}"
        position = random.choice(positions)
        skills_list = random.sample(skills, random.randint(1, len(skills)))
        skills_str = ", ".join(skills_list)
        performance_score = round(random.uniform(1, 5), 2)  # 绩效评分，1-5分
        training_record = random.choice(training_records)
        join_date = random_date(start_date, end_date).strftime("%Y-%m-%d")

        data.append({
            "姓名": name,
            "岗位": position,
            "能力标签": skills_str,
            "历史绩效": performance_score,
            "技能特长": skills_str,
            "培训记录": training_record,
            "入职日期": join_date
        })

    return pd.DataFrame(data)


# 生成数据集并保存为 CSV 文件
if __name__ == "__main__":
    dataset = generate_dataset(num_records=100)  # 可以调整生成的记录数量
    dataset.to_csv("人员数据集.csv", index=False, encoding="utf-8-sig")
    print("数据集已生成并保存为 '人员数据集.csv'")