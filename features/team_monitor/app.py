from flask import Flask, request, jsonify
import joblib
from train_models import *
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
# 加载模型
monitor_team = joblib.load('monitor_team_o1.pkl')


@app.route('/api/team/monitor', methods=['POST'])
def team_monitor():
    data = request.json
    team_id = data['teamId']
    current_staff = data['currentStaff']
    recent_changes = data['recentChanges']

    # 调用监控模型
    risk_warnings, optimization_suggestions = monitor_team(team_id, current_staff, recent_changes)

    # 生成返回数据
    response = {
        "code": 200,
        "message": "成功",
        "data": {
            "teamId": team_id,
            "riskWarnings": risk_warnings,
            "optimizationSuggestions": optimization_suggestions
        }
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True,port=5005)