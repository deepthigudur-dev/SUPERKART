import os
import requests
import pandas as pd
import streamlit as st

st.title("SuperKart Sales Prediction")
api_root = os.getenv("MODEL_API_URL", "http://superkart-backend:7860").rstrip("/")
st.subheader("Online Prediction")
Product_Weight = st.number_input("Product Weight", value=12.66)
Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Allocated Area", value=0.027, format="%.3f")
Product_MRP = st.number_input("Product MRP", value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age in Years", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Category", ["Perishables", "Non Perishables"])
product_data = {
    "Product_Weight": Product_Weight, "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area, "Product_MRP": Product_MRP,
    "Store_Size": Store_Size, "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type, "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years, "Product_Type_Category": Product_Type_Category
}
if st.button("Predict", type="primary"):
    try:
        response = requests.post(api_root + "/v1/predict", json=product_data, timeout=30)
        response.raise_for_status()
        st.success(f"Predicted sales: {response.json()['Prediction']:.2f}")
    except (requests.RequestException, KeyError, ValueError) as exc:
        st.error(str(exc))
st.subheader("Batch Prediction")
file = st.file_uploader("Upload CSV file", type=["csv"])
if file is not None and st.button("Predict for Batch"):
    try:
        response = requests.post(api_root + "/v1/predictbatch",
                                 files={"file": (file.name, file.getvalue(), "text/csv")}, timeout=60)
        response.raise_for_status()
        st.write(pd.DataFrame(list(response.json().items()), columns=["Row", "Prediction"]))
    except (requests.RequestException, ValueError) as exc:
        st.error(str(exc))
