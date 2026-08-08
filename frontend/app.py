
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

st.title("SuperKart Sales Prediction") #Complete the code to define the title of the app.

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0,value=0.05) #Complete the code to define the UI element for Product_Allocated_Area
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=150.0) #Complete the code to define the UI element for Product_MRP
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"]) #Complete the code to define the UI element for Store_Size
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]) #Complete the code to define the UI element for Store_Location_City_Type
Store_Type = st.selectbox("Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"]) #Complete the code to define the UI element for Store_Type
Product_Id_char = st.text_input("Product ID (e.g., FD6114)", value="FD6114") #Complete the code to define the UI element for Product_Id_char
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=10) #Complete the code to define the UI element for Store_Age_Years
Product_Type_Category = st.selectbox(
    "Product Type Category",
    ["Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
     "Health and Hygiene", "Baking Goods", "Bread", "Breakfast", "Frozen Foods",
     "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"]
)#Complete the code to define the UI element for Product_Type_Category

product_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}])

if st.button("Predict", type='primary'):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data.to_dict(orient='records')[0])
    if response.status_code == 200:
        result = response.json()
        predicted_sales = result["Sales"]
        st.write(f"Predicted Product Store Sales Total: ${predicted_sales:.2f}")
    else:
        st.error("Error in API request")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
