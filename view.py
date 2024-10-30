import streamlit as st
import joblib

# Tải mô hình và vectorizer
svm = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

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
