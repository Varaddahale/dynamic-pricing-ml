
import streamlit as st
import pickle
import numpy as np
import pandas as pd

rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
le_cat   = pickle.load(open("models/le_cat.pkl", "rb"))

st.set_page_config(page_title="Amazon Price Predictor")
st.title("Amazon Dynamic Price Predictor")
st.write("Enter product details to get the predicted price")

product_name  = st.text_input("Product name", "Boat USB Cable")
category      = st.selectbox("Category", le_cat.classes_)
discount_pct  = st.slider("Discount percentage (%)", 0, 90, 40)
rating        = st.slider("Rating", 1.0, 5.0, 4.0, step=0.1)
rating_count  = st.number_input("Number of ratings", 0, 500000, 1000)

name_length       = len(product_name)
category_encoded  = le_cat.transform([category])[0]
is_highly_rated   = 1 if rating >= 4.0 else 0
is_big_discount   = 1 if discount_pct >= 50 else 0
category_mean_price = 906.0

if st.button("Predict price"):
    features = [[
        discount_pct, rating, rating_count,
        name_length, category_encoded,
        category_mean_price,
        is_highly_rated, is_big_discount
    ]]
    log_pred = rf_model.predict(features)[0]
    price    = np.expm1(log_pred)
    st.success(f"Predicted price: Rs.{price:,.0f}")
    st.subheader("Input summary")
    st.write(pd.DataFrame({
        "Feature" : ["Category","Discount","Rating","Reviews"],
        "Value"   : [category, f"{discount_pct}%", rating, f"{rating_count:,}"]
    }))
