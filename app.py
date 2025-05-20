import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model pipeline
model = joblib.load('models/churn_model.pkl')

st.title("🔍 Customer Churn Prediction App")

# Option 1: Upload a CSV
st.subheader("📥 Upload a CSV file")
uploaded_file = st.file_uploader("Upload customer data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    preds = model.predict(df)
    df["Churn Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    st.write("📊 Prediction Results:")
    st.write(df)

# Option 2: Manual Input
st.subheader("🧍‍♀️ Predict for a Single Customer")

with st.form("single_customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiline = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bk = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", min_value=0.0)
    total = st.number_input("Total Charges", min_value=0.0)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiline,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bk,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mv,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")

# 📊 Data Visualization
st.subheader("📊 Customer Data Insights")

# Load a sample or uploaded dataset
data_source = uploaded_file if uploaded_file else "data/churn_data.csv"

try:
    df_viz = pd.read_csv(data_source)

    if 'Churn' in df_viz.columns:
        df_viz["Churn"] = df_viz["Churn"].map({"Yes": 1, "No": 0})

    st.markdown("### 1. Churn Count")
    churn_count = df_viz["Churn"].value_counts()
    st.bar_chart(churn_count)

    st.markdown("### 2. Churn by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x="gender", hue="Churn", data=df_viz, palette="Set2", ax=ax)
    st.pyplot(fig)

    st.markdown("### 3. Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df_viz, palette="Set1", ax=ax)
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.markdown("### 4. Monthly Charges Distribution by Churn")
    fig, ax = plt.subplots()
    sns.histplot(data=df_viz, x="MonthlyCharges", hue="Churn", bins=30, kde=True, ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.warning("📂 Please upload a dataset or make sure 'data/churn_data.csv' exists.")
    st.error(e)
