
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart Sales Predictor") #Complete the code to define the name of the app

# Load the trained churn prediction model
model = joblib.load("final_model.joblib") #Complete the code to define the location of the serialized model

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to SuperKart Sales Prediction API!" #Complete the code to define a welcome message

# Define an endpoint to predict churn for a single customer
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Extract relevant customer features from the input data. The order of the column names matters.
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a churn prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': prediction})

# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    Batch prediction endpoint for SuperKart Sales Forecasting.
    Accepts a CSV file, applies the same feature engineering used during training,
    selects the correct model input features, and returns predicted sales for each product.
    """

    # 1. Read uploaded CSV file
    file = request.files['file']
    saleskart_batch_data = pd.read_csv(file)

    # -----------------------------
    # 2. Feature Engineering (MUST match training notebook)
    # -----------------------------

    # Compute store age
    saleskart_batch_data['Store_Age_Years'] = 2024 - saleskart_batch_data['Store_Establishment_Year']

    # Extract first two characters of Product_Id
    saleskart_batch_data['Product_Id_char'] = saleskart_batch_data['Product_Id'].str[:2]

    # Map Product_Type to Perishables / Non‑Perishables (same as training)
    perishables = [
        "Fruits and Vegetables", "Meat", "Seafood", "Dairy",
        "Frozen Foods", "Baking Goods", "Bread"
    ]

    def change(x):
        return "Perishables" if x in perishables else "Non Perishables"

    saleskart_batch_data['Product_Type_Category'] = saleskart_batch_data['Product_Type'].apply(change)

    # -----------------------------
    # 3. Select final model input features
    # -----------------------------

    numeric_features = [
        'Product_Weight',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Age_Years'
    ]

    categorical_features = [
        'Product_Sugar_Content',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Product_Id_char',
        'Product_Type_Category'
    ]

    final_features = numeric_features + categorical_features

    batch_input_data = saleskart_batch_data[final_features]

    # -----------------------------
    # 4. Predict using trained model
    # -----------------------------

    predicted_sales = model.predict(batch_input_data).tolist()

    # -----------------------------
    # 5. Build output dictionary
    # -----------------------------

    product_ids = saleskart_batch_data['Product_Id'].tolist()
    output_dict = dict(zip(product_ids, predicted_sales))

    # -----------------------------
    # 6. Return JSON response
    # -----------------------------

    return jsonify(output_dict)

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
