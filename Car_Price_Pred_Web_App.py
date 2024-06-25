import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('lr_model.pkl')

data = pd.read_csv('car data.csv')
bike_names = [
    'Royal Enfield Thunder 500', 'UM Renegade Mojave', 'KTM RC200', 'Bajaj Dominar 400',
    'Royal Enfield Classic 350', 'KTM RC390', 'Hyosung GT250R', 'Royal Enfield Thunder 350',
    'KTM 390 Duke ', 'Mahindra Mojo XT300', 'Bajaj Pulsar RS200', 'Royal Enfield Bullet 350',
    'Royal Enfield Classic 500', 'Bajaj Avenger 220', 'Bajaj Avenger 150', 'Honda CB Hornet 160R',
    'Yamaha FZ S V 2.0', 'Yamaha FZ 16', 'TVS Apache RTR 160', 'Bajaj Pulsar 150',
    'Honda CBR 150', 'Hero Extreme', 'Bajaj Avenger 220 dtsi', 'Bajaj Avenger 150 street',
    'Yamaha FZ  v 2.0', 'Bajaj Pulsar  NS 200', 'Bajaj Pulsar 220 F', 'TVS Apache RTR 180',
    'Hero Passion X pro', 'Bajaj Pulsar NS 200', 'Yamaha Fazer ', 'Honda CB Trigger',
    'Yamaha FZ S ', 'Bajaj Pulsar 135 LS', 'Honda CB Unicorn', 'Hero Honda CBZ extreme',
    'Honda Karizma', 'TVS Jupyter', 'Hero Honda Passion Pro', 'Hero Splender Plus',
    'Honda CB Shine', 'Bajaj Discover 100', 'Hero Glamour', 'Hero Super Splendor',
    'Bajaj Discover 125', 'Hero Hunk', 'Hero  Ignitor Disc', 'Hero  CBZ Xtreme','Honda Activa 4G',
    'TVS Sport ','Honda Dream Yuga ','Bajaj Avenger Street 220','Hero Splender iSmart', 'Activa 3g', 'Hero Passion Pro',
       'Activa 4g', 'Honda Activa 125', 'Suzuki Access 125', 'TVS Wego',
       'Honda CB twister', 'Bajaj  ct 100'
]
data = data[~data["Car_Name"].isin(bike_names)]

# Extract unique values for categorical columns
car_names = data['Car_Name'].unique()
fuel_types = data['Fuel_Type'].unique()
selling_type = data["Selling_type"].unique()
transmissions = data['Transmission'].unique()
owners = data['Owner'].unique()

st.title("Car Price Prediction")

st.header("Input Car Features")
car_name = st.selectbox("Car Name", options=car_names)
year = st.number_input("Year", min_value=data['Year'].min(), max_value=data['Year'].max())
selling_price = st.number_input("Selling Price", min_value=0)
driven_kms = st.number_input("Driven Kilometers", min_value=0)
fuel = st.selectbox("Fuel Type", options=fuel_types)
selling = st.selectbox("Selling Type", options=selling_type)
transmission = st.selectbox("Transmission", options=transmissions)
owner = st.selectbox("Owner", options=owners)

input_data = {
    'Car_Name': [car_name],
    'Year': [year],
    'Selling_Price': [selling_price],
    'Driven_kms': [driven_kms],
    'Fuel_Type': [fuel],
    'Selling_type': [selling],
    'Transmission': [transmission],
    'Owner': [owner]
}

input_df = pd.DataFrame(input_data)

if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"The predicted price of the car is {prediction[0]:,.2f} Lakh INR")
