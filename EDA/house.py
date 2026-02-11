import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, encoders, and scaler
with open("house adaboost.pkl", "rb") as f:
    ada_model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f) 
    


st.title("🏠 Housing Price Prediction App")

# --- User Inputs ---
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=3)
size_in_sqft = st.number_input("Size in Sqft", min_value=300, max_value=10000, value=1500)
price_per_sqft = st.number_input("Price per Sqft", min_value=10, max_value=1000, value=80)
year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2018)
floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=5)
total_floors = st.number_input("Total Floors", min_value=1, max_value=200, value=10)
age_of_property = st.number_input("Age of Property (years)", min_value=0, max_value=100, value=5)
nearby_schools = st.text_input("Nearby Schools", "Delhi Public School, Whitefield")
nearby_hospitals = st.text_input("Nearby Hospitals", "Columbia Asia Hospital")
state = st.text_input("State", "Karnataka")
city = st.text_input("City", "Bangalore")
locality = st.text_input("Locality", "Whitefield")
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Plot"])
furnished_status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
public_transport_accessibility = st.text_input("Public Transport Accessibility", "Metro, Bus")
parking_space = st.selectbox("Parking Space", ["Yes", "No"])
security = st.text_input("Security", "24x7 Security")
amenities = st.text_input("Amenities", "Gym, Swimming Pool, Clubhouse")
facing = st.selectbox("Facing", ["East", "West", "North", "South"])
owner_type = st.selectbox("Owner Type", ["Individual", "Builder"])
availability_status = st.selectbox("Availability Status", ["Ready to Move", "Under Construction"])



# --- Prepare DataFrame ---
new_data = pd.DataFrame({
    'id':[1],
    'bhk':[bhk],
    'size_in_sqft':[size_in_sqft],
    'price_per_sqft':[price_per_sqft],
    'year_built':[year_built],
    'floor_no':[floor_no],
    'total_floors':[total_floors],
    'age_of_property':[age_of_property],
    'nearby_schools':[nearby_schools],
    'nearby_hospitals':[nearby_hospitals],
    'state':[state],
    'city':[city],
    'locality':[locality],
    'property_type':[property_type],
    'furnished_status':[furnished_status],
    'public_transport_accessibility':[public_transport_accessibility],
    'parking_space':[parking_space],
    'security':[security],
    'amenities':[amenities],
    'facing':[facing],
    'owner_type':[owner_type],
    'availability_status':[availability_status]
})


#label encoding
categorical_cols =new_data.select_dtypes(exclude=["int","float"])

for i in categorical_cols:
    new_data[i] = encoder.fit_transform(new_data[i])


# --- Prediction ---
if st.button("Predict Price"):
    #label encoding
    categorical_cols =new_data.select_dtypes(exclude=["int","float"])

    for i in categorical_cols:
        new_data[i] = encoder.fit_transform(new_data[i])
        
    #prediction
    prediction = ada_model.predict(new_data)
    base_price = float(prediction.item())  # convert to float

    st.success(f"Predicted Base Price: ₹{round(base_price, 2)}")

    # Future appreciation
    growth_rate = 0.06
    future_prices = {f"year {i}": base_price * ((1 + growth_rate) ** i) for i in range(1, 6)}

    st.subheader("📈 Future Prices (next 5 years)")
    st.write(pd.DataFrame(future_prices, index=["Price"]).T)

    # Optional chart
    st.line_chart(pd.DataFrame(future_prices, index=["Price"]).T)
