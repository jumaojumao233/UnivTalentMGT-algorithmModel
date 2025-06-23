from flask import Flask, request, jsonify
import joblib
from train_models import *
app = Flask(__name__)

# 加载模型
recommend_staff = joblib.load('recommend_staff.pkl')

@app.route('/api/team/recommend', methods=['POST'])
def team_recommend():
    data = request.json
    team_requirements = data['teamRequirements']

    # 调用推荐模型
    recommended_staff = recommend_staff(team_requirements)

    # 生成班子结构分析报告
    team_analysis = {
        'ageDistribution': 'Balanced',
        'genderRatio': '50:50',
        'skillCoverage': 'Complete'
    }

    return jsonify({
        "code": 200,
        "message": "成功",
        "data": {
            "recommendedStaff": recommended_staff,
            "teamAnalysis": team_analysis
        }
    })

if __name__ == '__main__':
    app.run(debug=True,port=5003)