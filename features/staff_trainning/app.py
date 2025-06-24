from flask import Flask, request, jsonify
import joblib
from train_models import *
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
# 加载模型
recommend_training = joblib.load('recommend_training.pkl')


@app.route('/api/training/recommend', methods=['POST'])
def training_recommend():
    data = request.json
    staff_id = data['staffId']
    capabilities = data['capabilities']
    position_requirements = data['positionRequirements']

    # 调用推荐模型
    training_needs, training_plan = recommend_training(staff_id, capabilities, position_requirements)

    # 生成返回数据
    response = {
        "code": 200,
        "message": "成功",
        "data": {
            "staffId": staff_id,
            "trainingNeeds": [{"skill": skill, "priority": "High"} for skill in training_needs],
            "trainingPlan": training_plan
        }
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True,port=5004)