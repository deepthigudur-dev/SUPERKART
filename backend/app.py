import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Predictor")
model = joblib.load("superkart_model.joblib")
feature_columns = [
    "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years",
    "Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
    "Store_Type", "Product_Id_char", "Product_Type_Category"
]

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@superkart_api.post("/v1/predict")
def predict_sales():
    # Get JSON data from the request
    product_data = request.get_json()

    # Extract relevant product and store features
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_char': product_data['Product_Id_char'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }
    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a churn prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]
  
    return jsonify({"Prediction": prediction})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
   # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)
    
    predictions = model.predict(input_data[feature_columns]).tolist()
    return jsonify(dict(zip(map(str, range(len(predictions))), predictions)))

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
