import streamlit as st

# Giao diện Streamlit
st.title("Spam Classifier")
input_message = st.text_area("Nhập tin nhắn của bạn:")

if st.button("Dự đoán"):
    input_vector = vectorizer.transform([input_message]).toarray()
    
    log_reg_pred = log_reg.predict(input_vector)
    svm_pred = svm.predict(input_vector)
    nb_pred = nb.predict(input_vector)

    result = {
        'Logistic Regression': 'Spam' if log_reg_pred[0] == 1 else 'Ham',
        'SVM': 'Spam' if svm_pred[0] == 1 else 'Ham',
        'Naive Bayes': 'Spam' if nb_pred[0] == 1 else 'Ham',
    }

    st.write(result)
