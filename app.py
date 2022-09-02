import streamlit as st #pip install streamlit to make website
import pickle #pip install pickle
from sklearn.feature_extraction.text import TfidfVectorizer #vectorize data
import string #string library

tfidf = pickle.load(open('vectorizer.pkl', 'rb')) #load vectorizer
model = pickle.load(open('model.pkl', 'rb')) #load model

st.title('Spam Classifier') #title

input = st.text_area('Enter your message here') #text area
input = [input] #convert to list




if st.button('Classify'): #button

    
    feature_extractor= TfidfVectorizer(min_df=1,stop_words='english',lowercase='True') #create vectorizer
     
    final_data = feature_extractor.fit_transform(input) #fit vectorizer

    predictions = model.predict(final_data) #predict

    if (predictions[0]==0): #if else statement to classify message
        st.header("The mail received is NOT SPAM")
    else:
        st.header("The mail received is SPAM")