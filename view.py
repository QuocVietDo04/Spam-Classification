import streamlit as st
import re
import numpy as np
from collections import defaultdict
import math

# Bước 1: Tiền xử lý văn bản
stop_words = set(["the", "is", "in", "and", "to", "of", "for", "a", "an", "on"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return words

# Bước 2: Tạo model Naive Bayes
def train_naive_bayes(spam_texts, ham_texts):
    def calculate_word_frequencies(texts):
        word_counts = defaultdict(int)
        total_words = 0
        for text in texts:
            for word in preprocess_text(text):
                word_counts[word] += 1
                total_words += 1
        return word_counts, total_words

    spam_counts, spam_total = calculate_word_frequencies(spam_texts)
    ham_counts, ham_total = calculate_word_frequencies(ham_texts)
    vocab = set(spam_counts.keys()).union(set(ham_counts.keys()))

    p_spam = len(spam_texts) / (len(spam_texts) + len(ham_texts))
    p_ham = len(ham_texts) / (len(spam_texts) + len(ham_texts))

    return {
        'spam_counts': spam_counts,
        'ham_counts': ham_counts,
        'spam_total': spam_total,
        'ham_total': ham_total,
        'p_spam': p_spam,
        'p_ham': p_ham,
        'vocab': vocab
    }

# Bước 3: Dự đoán với model
def predict_naive_bayes(model, text):
    spam_counts = model['spam_counts']
    ham_counts = model['ham_counts']
    spam_total = model['spam_total']
    ham_total = model['ham_total']
    p_spam = model['p_spam']
    p_ham = model['p_ham']
    vocab = model['vocab']
    vocab_size = len(vocab)

    words = preprocess_text(text)
    spam_prob = math.log(p_spam)
    ham_prob = math.log(p_ham)

    for word in words:
        spam_word_prob = (spam_counts[word] + 1) / (spam_total + vocab_size)
        ham_word_prob = (ham_counts[word] + 1) / (ham_total + vocab_size)
        spam_prob += math.log(spam_word_prob)
        ham_prob += math.log(ham_word_prob)

    return "Spam" if spam_prob > ham_prob else "Ham"

# Tạo giao diện Streamlit
st.title("Spam Classifier")
st.write("Nhập nội dung tin nhắn hoặc email để kiểm tra xem có phải thư rác không:")

# Tạo dữ liệu huấn luyện mẫu
spam_texts = ["win big money now", "click here to claim your prize", "limited time offer", "free entry in a contest"]
ham_texts = ["let's meet for lunch tomorrow", "can you send the report?", "please call me when you're free"]

# Huấn luyện mô hình
model = train_naive_bayes(spam_texts, ham_texts)

# Lấy đầu vào từ người dùng
user_input = st.text_area("Nhập nội dung ở đây:")

if st.button("Dự đoán"):
    prediction = predict_naive_bayes(model, user_input)
    st.write(f"Kết quả: **{prediction}**")
