from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    
model = load_model('random_forest_regressor')


st.title('Temperature Prediction App')
st.write('This web app is based on solving a regression problem which is to predict the Temperaturebased on several features that you can see in the sidebar. Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction of the Temperature.')


# In[22]:


#these are the numeric attributes fed to the model

apparent_temperature = st.sidebar.slider(label = 'Apparent Temperature (C)', min_value = 4.0,
                          max_value = 16.0 ,
                          value = 10.0,
                          step = 0.1)

humidity = st.sidebar.slider(label = 'Humidity', min_value = 0.00,
                          max_value = 2.00 ,
                          value = 1.00,
                          step = 0.01)

month = st.sidebar.slider(label = 'Month', min_value = 1,
                          max_value = 12 ,
                          value = 1,
                          step = 1) 

year = st.sidebar.slider(label = 'Year', min_value = 2006,
                          max_value = 2021 ,
                          value =2009,
                          step =1) 

hour = st.sidebar.slider(label = 'hour', min_value = 0,
                          max_value = 23 ,
                          value =1,
                          step =1)

day = st.sidebar.slider(label = 'day', min_value = 1,
                          max_value = 31 ,
                          value =1,
                          step =1)


wind_speed = st.sidebar.slider(label = 'Wind Speed (km/h)', min_value = 10.0,
                          max_value = 100.0 ,
                          value = 0.50,
                          step = 0.01) 

wind_bearing = st.sidebar.slider(label = 'Wind Bearing (degrees)', min_value = 0,
                          max_value = 360 ,
                          value = 8,
                          step = 1)

visibility = st.sidebar.slider(label = 'Visibility (km)', min_value = 0.0,
                          max_value = 20.000 ,
                          value = 0.500,
                          step = 0.5)
   
pressure = st.sidebar.slider(label = 'Pressure (millibars)', min_value = 0.0,
                          max_value = 1050.0,
                          value = 100.0,
                          step = 10.0)

#these are the categorical attributes fed to the model

summmary = st.radio("Summary", ['Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Foggy',
       'Breezy and Mostly Cloudy', 'Clear', 'Breezy and Partly Cloudy',
       'Breezy and Overcast', 'Humid and Mostly Cloudy',
       'Humid and Partly Cloudy', 'Windy and Foggy', 'Windy and Overcast',
       'Breezy and Foggy', 'Windy and Partly Cloudy', 'Breezy',
       'Dry and Partly Cloudy', 'Windy and Mostly Cloudy',
       'Dangerously Windy and Partly Cloudy', 'Dry', 'Windy',
       'Humid and Overcast', 'Light Rain', 'Drizzle', 'Windy and Dry',
       'Dry and Mostly Cloudy', 'Breezy and Dry', 'Rain'])

prec_type = st.radio("Precip Type", ['rain', 'snow'])


# In[24]:


features = {'Apparent Temperature (C)': apparent_temperature, 'Humidity': humidity,
            'Wind Speed (km/h)': wind_speed, 'Wind Bearing (degrees)': wind_bearing,
            'Visibility (km)': visibility, 'Pressure (millibars)': pressure,
            'day': day, 'hour': hour,
            'Month': month, 'year': year, 'Precip Type': prec_type, 'Summary':summmary
            }
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_quality(model, features_df)
    
    st.write(' Based on feature values, Temperature is '+ str(prediction)+ " °C")


# In[ ]:




