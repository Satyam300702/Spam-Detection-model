# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 06:46:02 2025

@author: HP
"""

import os
import pickle
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__),"spam_model.sav")
vectorizer_path = os.path.join(os.path.dirname(__file__),"tfidf_vectorizer.pkl")

try:
    model = pickle.load(open(model_path,"rb"))
    vectorizer = pickle.load(open(vectorizer_path,"rb"))
except FileNotFoundError:
    st.error("Model File not found")
    st.stop()
    
def spam_prediction(message):
    transformed_input = vectorizer.transform([message])
    prediction = model.predict(transformed_input)[0]
    proability = model.predict_proba(transformed_input)[0][prediction]
    
    if prediction == 0:
        return "This message is Spam"
    else:
        return "This message is Not Spam(Ham)"
    
def main():
    st.title("Spam mail Detection App")
    st.write("This app uses a **Stacking Ensemble Model** (Naive Bayes + Gradient Boosting + Logistic Regression) to classify messages as spam or not spam.")
    user_input = st.text_area("type or paste the message text:",height=200)
    
    result = ''
    if st.button("Predict"):
        if user_input.strip()=="":
            st.warning("Please enter a message befor predicting")
        else:
            result = spam_prediction(user_input)
            st.success(result)
if __name__ == "__main__":
    main()

        