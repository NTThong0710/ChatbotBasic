from flask import Flask, request, jsonify, render_template
import pandas as pd
import unicodedata
import re
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Tải dữ liệu từ file CSV
csv_file = "DATA_CHATBOT.csv"
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file).dropna()  # Loại bỏ các hàng trống
else:
    raise FileNotFoundError(f"Tệp '{csv_file}' không tồn tại.")

# Hàm loại bỏ dấu tiếng Việt
def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return text.lower()

# Tạo từ điển cho phản hồi
responses = {remove_accents(row["CÂU HỎI"]): row["CÂU TRẢ LỜI"] for _, row in data.iterrows()}

# Tạo TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Hàm lấy phản hồi từ từ điển với độ tương đồng
def get_response(user_input):
    normalized_input = remove_accents(user_input)
    
    # Tạo một danh sách câu hỏi từ cơ sở dữ liệu
    questions = list(responses.keys())
    questions.append(normalized_input)  # Thêm câu hỏi của người dùng vào danh sách

    # Tính toán TF-IDF cho các câu hỏi
    tfidf_matrix = vectorizer.fit_transform(questions)

    # Tính toán độ tương đồng giữa câu hỏi người dùng và các câu hỏi trong cơ sở dữ liệu
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Tìm câu hỏi có độ tương đồng cao nhất
    most_similar_index = similarities.argmax()
    most_similar_question = questions[most_similar_index]

    return responses.get(most_similar_question, "Xin lỗi, tôi không hiểu câu hỏi của bạn.")

# Hàm lưu lịch sử cuộc trò chuyện vào file JSON
def save_chat_history(user_input, bot_response):
    history_file = "chat_history.json"
    # Kiểm tra nếu file đã tồn tại thì đọc nội dung hiện có
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as file:
            chat_history = json.load(file)
    else:
        chat_history = []

    # Thêm tin nhắn mới vào lịch sử
    chat_history.append({"user": user_input, "bot": bot_response})

    # Ghi lại lịch sử vào file JSON
    with open(history_file, "w", encoding="utf-8") as file:
        json.dump(chat_history, file, ensure_ascii=False, indent=4)

# Endpoint để xử lý yêu cầu của người dùng
@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("message", "")
    response = get_response(user_input)
    save_chat_history(user_input, response)  # Lưu tin nhắn vào lịch sử
    return jsonify({"response": response})

# Trang HTML chính cho chatbot
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat_history.json')
def get_chat_history():
    # Đọc dữ liệu từ file chat_history.json
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r', encoding="utf-8") as f:
            chat_history = json.load(f)
    else:
        chat_history = []  # Nếu file không tồn tại, trả về danh sách rỗng
    return jsonify(chat_history)

@app.route('/clear_history', methods=['POST'])
def clear_chat_history():
    # Xóa nội dung trong file chat_history.json
    with open('chat_history.json', 'w', encoding="utf-8") as f:
        json.dump([], f)  # Lưu một danh sách rỗng để xóa tất cả
    return jsonify({})  # Không trả về thông báo gì
@app.route("/faq_questions")
def faq_questions():
    faq_list = list(responses.keys())  # Lấy các câu hỏi từ từ điển
    return jsonify(faq_list)

if __name__ == "__main__":
    app.run(debug=True)
