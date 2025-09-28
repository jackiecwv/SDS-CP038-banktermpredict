import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Load model and preprocessing objects ---
@st.cache_resource
def load_artifacts():
    model = load_model('focal_model.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)})
    scaler = joblib.load('scaler.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    return model, scaler, label_encoders

    drop_features = joblib.load('drop_features.joblib')  # List of features dropped due to VIF
    return model, scaler, label_encoders, drop_features
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = alpha * tf.pow(1. - pt, gamma) * bce
        return tf.reduce_mean(loss)
    'campaign_intensity', 'job_education', 'married_with_loan', 'single_with_housing', 'recent_contact'
model, scaler, label_encoders = load_artifacts()

# --- Feature list (update to match your model) ---
feature_list = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
    'campaign_intensity', 'job_education', 'married_with_loan', 'single_with_housing', 'recent_contact'
]

# --- Streamlit UI ---
st.title("Bank Term Deposit Prediction (Focal Loss Model)")

st.write("Enter customer details to predict the likelihood of subscribing to a term deposit.")

# --- Input form for all features ---
user_input = {}
for feature in feature_list:
    if feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        # Use selectbox for categorical features
        options = label_encoders[feature].classes_
        user_input[feature] = st.selectbox(f"{feature.capitalize()}", options)
    else:
        # Use number input for numeric features
        user_input[feature] = st.number_input(f"{feature.capitalize()}", value=0)

# --- Feature engineering (must match notebook) ---
def engineer_features(df):
    # Example: update to match your notebook's feature engineering
    df['campaign_intensity'] = pd.cut(df['campaign'], bins=[-1, 2, 5, df['campaign'].max()], labels=['low', 'medium', 'high'])
    df['job_education'] = df['job'] + '_' + df['education']
    df['married_with_loan'] = ((df['marital'] == 'married') & (df['loan'] == 'yes')).astype(int)
    df['single_with_housing'] = ((df['marital'] == 'single') & (df['housing'] == 'yes')).astype(int)
    df['recent_contact'] = (df['pdays'] < 30).astype(int)
    return df

# --- Predict button ---
if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    # Feature engineering
    input_df = engineer_features(input_df)
    # Encode categorical features
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])
    # Scale numeric features
    numeric_cols = scaler.feature_names_in_
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    # Predict
    pred_prob = model.predict(input_df)[0][0]
    pred = int(pred_prob > 0.5)
    st.write(f"**Prediction:** {'Subscribed' if pred else 'Not Subscribed'} (Probability: {pred_prob:.2f})")