from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
from flask_cors import CORS
CORS(app)
# 加载模型
capability_model = joblib.load('capability_model.pkl')
potential_model = joblib.load('potential_model.pkl')
development_model = joblib.load('development_model.pkl')


@app.route('/api/assessment/tags', methods=['POST'])
def generate_tags():
    data = request.json
    staff_id = data['staffId']
    capabilities = data['capabilities']
    additional_info = data['additionalInfo']

    # 提取特征
    input_features = [
        capabilities['teaching'],
        capabilities['research'],
        capabilities['management'],
        capabilities['innovation'],
        additional_info['yearsOfExperience']
    ]

    # 预测标签
    capability_tag = capability_model.predict([input_features])[0]
    potential_tag = potential_model.predict([input_features])[0]
    development_tag = development_model.predict([input_features])[0]

    # 生成标签列表
    tags = [capability_tag, potential_tag, development_tag]

    return jsonify({
        "code": 200,
        "message": "成功",
        "data": {
            "staffId": staff_id,
            "tags": tags
        }
    })


if __name__ == '__main__':
    app.run(debug=True,port=5002)