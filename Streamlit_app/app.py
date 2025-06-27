import streamlit as st
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import json
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"  # Change if your FastAPI runs elsewhere

CAR_NAMES = [
    'ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza',
    's cross', 'alto 800', 'ertiga', 'dzire', 'alto k10', 'ignis',
    '800', 'baleno', 'omni', 'fortuner', 'innova', 'corolla altis',
    'etios cross', 'etios g', 'etios liva', 'corolla', 'etios gd',
    'camry', 'land cruiser', 'i20', 'grand i10', 'i10', 'eon', 'xcent',
    'elantra', 'creta', 'verna', 'city', 'brio', 'amaze', 'jazz'
]

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction")

# Sidebar for inputs
st.sidebar.header("Input Car Features")

car_name = st.sidebar.selectbox("Car Name", options=CAR_NAMES)

# Year with current year as max
from datetime import datetime
current_year = datetime.now().year
year = st.sidebar.number_input("Year", min_value=1900, max_value=current_year, value=2015)

# Better labeling and validation for price
present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

# Driven KMs with better default
driven_kms = st.sidebar.number_input("Driven KMs", min_value=0, max_value=1000000, value=30000, step=1000)

# Dropdown options
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
selling_type = st.sidebar.selectbox("Selling Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# Owner with better labels
owner_options = {
    "First Owner": 0,
    "Second Owner": 1,
    "Third Owner": 2,
    "Fourth & Above Owner": 3
}
owner_label = st.sidebar.selectbox("Owner", list(owner_options.keys()))
owner = owner_options[owner_label]

# Validation
if st.sidebar.button("Validate Inputs"):
    if not car_name.strip():
        st.sidebar.error("Please enter a car name")
    elif year > current_year:
        st.sidebar.error(f"Year cannot be greater than {current_year}")
    elif present_price <= 0:
        st.sidebar.error("Present price must be greater than 0")
    elif driven_kms < 0:
        st.sidebar.error("Driven KMs cannot be negative")
    else:
        st.sidebar.success("All inputs are valid!")

# Prepare input data
input_data = {
    "Car_Name": car_name.strip(),
    "Year": int(year),
    "Present_Price": float(present_price),
    "Driven_kms": int(driven_kms),
    "Fuel_Type": fuel_type,
    "Selling_type": selling_type,
    "Transmission": transmission,
    "Owner": int(owner)
}

# Display input data
st.subheader("📋 Input Summary")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Car Name:** {input_data['Car_Name']}")
    st.write(f"**Year:** {input_data['Year']}")
    st.write(f"**Present Price:** ₹{input_data['Present_Price']:.2f} lakhs")
    st.write(f"**Driven KMs:** {input_data['Driven_kms']:,}")

with col2:
    st.write(f"**Fuel Type:** {input_data['Fuel_Type']}")
    st.write(f"**Selling Type:** {input_data['Selling_type']}")
    st.write(f"**Transmission:** {input_data['Transmission']}")
    st.write(f"**Owner:** {owner_label}")

# Prediction buttons
st.subheader("🔮 Predictions")
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🎯 Predict Best Model", use_container_width=True)

# Helper function for API calls
def make_api_request(endpoint, data):
    """Make API request with proper error handling"""
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}", 
            json=data, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json(), None
    except ConnectionError:
        return None, "❌ Cannot connect to the API server. Please check if the FastAPI server is running."
    except Timeout:
        return None, "⏱️ Request timed out. The server might be busy."
    except requests.exceptions.HTTPError as e:
        if response.status_code == 422:
            return None, f"❌ Invalid input data: {response.text}"
        else:
            return None, f"❌ Server error ({response.status_code}): {response.text}"
    except RequestException as e:
        return None, f"❌ Request failed: {str(e)}"
    except json.JSONDecodeError:
        return None, "❌ Invalid response format from server"

# Validation before making requests
def validate_inputs():
    """Validate inputs before making API calls"""
    errors = []
    
    if not input_data['Car_Name'].strip():
        errors.append("Car name is required")
    
    if input_data['Year'] > current_year:
        errors.append(f"Year cannot be greater than {current_year}")

    if input_data['Present_Price'] <= 0:
        errors.append("Present price must be greater than 0")

    if input_data['Driven_kms'] < 0:
        errors.append("Driven KMs cannot be negative")
    
    return errors

# Handle best model prediction
if predict_btn:
    validation_errors = validate_inputs()
    
    if validation_errors:
        st.error("Please fix the following issues:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        with st.spinner("🔍 Finding the best model..."):
            result, error = make_api_request("predict/best", input_data)
            
            if error:
                st.error(error)
            elif result:
                st.success("✅ Prediction completed!")
                
                # Display best model result
                st.subheader("🏆 Best Model Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Best Model",
                        value=result['best_model'].replace('_', ' ').title()
                    )
                
                with col2:
                    st.metric(
                        label="Predicted Price",
                        value=f"₹{result['predicted_price']:.2f} lakhs"
                    )
                
                with col3:
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']*100:.2f}%"
                    )
                
                # Show all results in expandable section
                with st.expander("📊 All Model Results"):
                    for model, res in result["all_results"].items():
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**{model.replace('_', ' ').title()}**")
                        with col2:
                            st.write(f"₹{res['prediction']:.2f} lakhs ({res['confidence']*100:.2f}%)")

# Footer
st.markdown("---")
st.markdown("*Make sure your FastAPI server is running on the specified URL before making predictions.*")