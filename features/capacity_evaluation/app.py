from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
models = {}
for target in ['teaching', 'research', 'management', 'innovation']:
    models[target] = joblib.load(f'{target}_model.pkl')


@app.route('/api/assessment/capability', methods=['POST'])
def assess_capability():
    data = request.json
    staff_id = data['staffId']
    features = data['features']

    # 提取特征
    input_features = [
        features['teaching']['courseCount'],
        features['teaching']['studentEvaluation'],
        features['teaching']['awards'],
        features['research']['publications'],
        features['research']['patents'],
        features['research']['projectFunding'],
        features['management']['teamSize'],
        features['management']['collaborations'],
        features['innovation']['innovativeProjects'],
        # features['innovation']['awards']
    ]

    # 预测能力评分
    predictions = {target: model.predict([input_features])[0] for target, model in models.items()}

    return jsonify({
        "code": 200,
        "message": "成功",
        "data": {
            "staffId": staff_id,
            "capabilities": predictions
        }
    })


if __name__ == '__main__':
    app.run(debug=True,port=5001)