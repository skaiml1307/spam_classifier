import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps=PorterStemmer()

tfidf= pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    L=[]
    for i in text:
        if i.isalnum():
            L.append(i)
    
    text=L.copy()
    L.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            L.append(i)

    text=L.copy()
    L.clear()
    for i in text:
        L.append(ps.stem(i))
        
    return " ".join(L)

st.title("Email/SMS Spam Classifier")
input_sms= st.text_area("Enter the message")

if st.button('Predict'):
    # 1. text preprocess
    transformed_sms= transform_text(input_sms)
    # 2. vectorize
    vector_input= tfidf.transform([transformed_sms])
    # 3. predict
    result= model.predict(vector_input)[0]
    # 4. display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")
