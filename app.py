from flask import Flask, request, jsonify, render_template
import pandas as pd
import unicodedata
import re
import os

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

# Hàm lấy phản hồi từ từ điển
def get_response(user_input):
    normalized_input = remove_accents(user_input)
    return responses.get(normalized_input, "Xin lỗi, tôi không hiểu câu hỏi của bạn.")

# Endpoint để xử lý yêu cầu của người dùng
@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_input = request.json["message"]
    response = get_response(user_input)
    return jsonify({"response": response})

# Trang HTML chính cho chatbot
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
