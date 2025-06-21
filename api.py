from flask import Flask, request, jsonify, Response, stream_with_context
import pandas as pd
import joblib
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
data = pd.read_csv("aboutDataset/人员数据集.csv")
# 点赞
@app.route('/word/like/create', methods=['POST'])
def like_post():
    try:
        data = request.get_json()
        post_id = data.get('postId')
        phone = data.get('phone')

        if not post_id or not phone:
            return jsonify({'error': '缺少参数'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # 检查是否已经点过赞
        check_query = "SELECT COUNT(*) AS count FROM likes WHERE post_id=%s AND user_id=%s"
        cursor.execute(check_query, (post_id, phone))
        result = cursor.fetchone()

        if result['count'] > 0:
            conn.close()
            return jsonify({'message': '已点赞'}), 400

        # 插入新的点赞记录
        insert_query = "INSERT INTO likes (post_id, user_id,liked) VALUES (%s, %s,1)"
        cursor.execute(insert_query, (post_id, phone))
        update_query = """
            UPDATE
            posts
            SET
            like_count = like_count + 1
            WHERE
            id = %s;
            """
        cursor.execute(update_query, (int(post_id),))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': '点赞成功'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    形如{'岗位': '教学岗位', '能力标签': '教学设计, 团队协作, 数据分析, 领导力, 科研能力',
                  '技能特长': '教学设计, 团队协作, 数据分析, 领导力, 科研能力', '培训记录': '课程B', '入职年份': 2021,
                  '入职月份': 12}
    to
    :return:
    """
    # 直接使用预训练模型进行预测
    try:
        input_json = request.json
        modelName = 'decission_tree'
        label_encoders=joblib.load('aboutModel/saved_stander/'+modelName+'.pkl')
        for key in input_json:
            input_json[key] = label_encoders[key].transform([input_json[key]])
        model = joblib.load('aboutModel/saved_model/'+modelName+'.pkl')
        preds = "否"
        data_input = {}
        for i in input_json:
            data_input[i] = input_json[i][0]
        if int(model.predict([list(data_input.values())])):
            preds = "是"
        print(preds)
        return jsonify({
            "prediction":preds
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # app.run(debug=True,host='192.168.100.52',port=3000)
    app.dt_model=None
    
    app.run(debug=True, port=3000)
