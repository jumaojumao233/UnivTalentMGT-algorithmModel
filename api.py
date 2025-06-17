from flask import Flask, request, jsonify, Response, stream_with_context

app = Flask(__name__)
from flask_cors import CORS
CORS(app)
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
    return jsonify({
        "mesg":"hello predict"
    })


if __name__ == '__main__':
    # app.run(debug=True,host='192.168.100.52',port=3000)
    app.run(debug=True, port=3000)
