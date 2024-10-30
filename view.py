import streamlit as st
import pickle

# Tải mô hình và vectorizer
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Giao diện Streamlit
st.title("Spam Classifier")
input_message = st.text_area("Nhập tin nhắn của bạn:")

if st.button("Dự đoán"):
    # Chuyển đổi đầu vào của người dùng thành vector
    input_vector = vectorizer.transform([input_message]).toarray()
    
    # Dự đoán bằng mô hình SVM
    svm_pred = svm.predict(input_vector)

    # Kết quả dự đoán
    result = 'Spam' if svm_pred[0] == 1 else 'Ham'
    
    st.write(f"Kết quả: {result}")
# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split

# # Dữ liệu mẫu
# data = {
#     'text': [
#         'Free money now!!!',
#         'Hi, how are you?',
#         'Exclusive deal just for you!',
#         'See you at the meeting tomorrow.',
#         'Congratulations! You have won a lottery.',
#         'Important notice about your account.'
#     ],
#     'label': [1, 0, 1, 0, 1, 0]  # 1: Spam, 0: Ham
# }

# # Chuyển đổi dữ liệu thành DataFrame
# df = pd.DataFrame(data)

# # Tách dữ liệu thành tập huấn luyện và kiểm tra
# X = df['text']
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Xây dựng mô hình SVM
# model = make_pipeline(CountVectorizer(), SVC(probability=True))
# model.fit(X_train, y_train)

# # Giao diện Streamlit
# st.title("Spam Classifier")
# input_message = st.text_area("Nhập tin nhắn của bạn:")

# if st.button("Dự đoán"):
#     # Dự đoán bằng mô hình SVM
#     prediction = model.predict([input_message])
#     prediction_proba = model.predict_proba([input_message])

#     # Kết quả dự đoán
#     result = 'Spam' if prediction[0] == 1 else 'Ham'
#     proba = np.max(prediction_proba) * 100  # Tính xác suất

#     st.write(f"Kết quả: {result} (Xác suất: {proba:.2f}%)")
