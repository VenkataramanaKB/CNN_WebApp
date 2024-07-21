import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model 
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt


model = load_model('assets/new.h5', compile=False)
st.title('High Performance Liquid Chromotography Prediction')
uploadedFile = st.file_uploader('Upload your CSV', type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")

if uploadedFile is not None:
    data = pd.read_csv(uploadedFile)
    arr = data.columns
    
    X= data.drop(columns=['Sample_ID','Sample_ID_2'],axis=1)
    result= model.predict(X)
    average= [sum(x)/len(x) for x in zip(*result)]
    st.write("The Predicted HPLC is")
    st.write(average)


