import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
from keras.models import load_model
import streamlit as slt
from sklearn.preprocessing import MinMaxScaler

import datetime as dt
import yfinance as yf
start = '2010-01-01'
end = '2022-01-01'



slt.title("Stock Price Prediction")
user_input=slt.text_input('Enter Stock Ticker','AAPL')
df = yf.Ticker(user_input)
df = df.history(start=start, end=end) 

# df=data.DataReader(user_input,'yahoo',start,end)

# Describing The Data

slt.subheader("Data from 2010 - 2021")
slt.write(df.describe())

# Visualisation

slt.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
slt.pyplot(fig)


slt.subheader("Closing Price vs Time Chart with 100MA")
fig=plt.figure(figsize=(12,6))
ma100=df.Close.rolling(100).mean()
plt.plot(df.Close)
plt.plot(ma100,'r')
slt.pyplot(fig)


slt.subheader("Closing Price vs Time Chart with 100MA and 200MA")
fig=plt.figure(figsize=(12,6))
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
slt.pyplot(fig)


# Splitting Data into training and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)



# Load My Model
model=load_model('keras_model.h5')
# testing Part
past_100_days=data_training.tail(100)
final_df=past_100_days._append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df) 
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):

    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
my_scaler=scaler.scale_
scale_factor=1/my_scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


# Final Graph

slt.subheader("Prediction vs Orignal")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Orignal Price')
plt.plot(y_predicted,'r',label='Pridicted Price')
plt.xlabel("Time")
plt.ylabel('price')
plt.legend()
slt.pyplot(fig2)