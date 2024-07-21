import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model 
from sklearn.model_selection import  train_test_split



model = load_model('assets/new.h5', compile=False)
st.title('API Content Prediction using NIR')
uploadedFile = st.file_uploader('Upload your CSV', type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
labels = ['active','Sample_ID', 'Sample_ID_2', 'Target_HPLC']
if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)
    arr = data.columns
    
    if 'active' in arr:
        X= data.drop(columns=['active','Sample_ID','Sample_ID_2'],axis=1)
        
    elif 'Target_HPLC' in arr:
        X= data.drop(columns=['Target_HPLC','Sample_ID','Sample_ID_2'],axis=1)
        
    else:
        X= data.drop(columns=['Sample_ID','Sample_ID_2'],axis=1)
    st.scatter_chart(X)
    result= model.predict(X)
    average= [sum(x)/len(x) for x in zip(*result)]
    st.write("The Predicted HPLC is")
    st.write(average)


