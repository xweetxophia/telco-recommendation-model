import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pypickle


#load the model
loaded_model = pypickle.load('tele.pkl')

#create a function that called prediction that will take in functions entered by the users

def prediction(data):
    # Create a DataFrame from the input data
    df = pd.DataFrame([data], columns=[
       'customerID' ,'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ])

    # Convert the data types of the columns
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    df['tenure'] = df['tenure'].astype(int)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Encode categorical columns
    cat_cols = ['customerID' ,'gender', 'Partner', 'Dependents', 'PhoneService', 
                'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod']
    
    label = LabelEncoder()
    for col in cat_cols:
        df[col] = label.fit_transform(df[col])


    # Prepare data for prediction
    num_data = df.values.reshape(1, -1)
    scalar = StandardScaler()
    num_data = scalar.fit_transform(num_data)

    # Predicting the model
    pred = loaded_model.predict(num_data)

    if pred[0] == 0:
        return "The customer is likely to stay"
    else:
        return "The customer is likely to churn"

def main():
    st.image("Grey Modern.png")
    st.title("Telco Customer Churn Prediction")
    customerID = st.text_input("customerID: ")
    gender = st.selectbox("Gender: ", ('Female', 'Male'))
    SeniorCitizen = st.radio("Are you a senior citizen? ", (0, 1))
    Partner = st.radio("Do you have a partner? ", ('Yes', 'No'))
    Dependents = st.radio("Do you have dependents? ", ('Yes', 'No'))
    tenure = st.number_input("Tenure (months): ", min_value=0)
    PhoneService = st.radio("Phone service? ", ('Yes', 'No'))
    MultipleLines = st.radio("Multiple lines? ", ('Yes', 'No', 'No phone service'))
    InternetService = st.selectbox("Internet service type: ", ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.radio("Online security? ", ('Yes', 'No', 'No internet service'))
    OnlineBackup = st.radio("Online backup? ", ('Yes', 'No', 'No internet service'))
    DeviceProtection = st.radio("Device protection? ", ('Yes', 'No', 'No internet service'))
    TechSupport = st.radio("Tech support? ", ('Yes', 'No', 'No internet service'))
    StreamingTV = st.radio("Streaming TV? ", ('Yes', 'No', 'No internet service'))
    StreamingMovies = st.radio("Streaming movies? ", ('Yes', 'No', 'No internet service'))
    Contract = st.selectbox("Contract type: ", ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.radio("Paperless billing? ", ('Yes', 'No'))
    PaymentMethod = st.selectbox("Payment method: ", ('Electronic check', 'Mailed check', 
                                                      'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.number_input("Monthly charges: ", min_value=0.0)
    TotalCharges = st.text_input("Total charges: ")

    if st.button("Predict"):
        result = prediction([customerID,gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
                            MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
                            DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
                            Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges])
        st.success(result)

if __name__ == "__main__":
    main()
