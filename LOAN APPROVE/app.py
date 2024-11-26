import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Load the best model and scaler
try:
    model = pickle.load(open("models/best_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Please run train_multiple_models.py first to generate the model files!")
    st.stop()

def predict_loan_approval(data):
    # Scale the input data
    scaled_data = scaler.transform(data)
    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    return prediction[0], probability[0][1]

def main():
    st.title("Loan Approval Prediction System")
    
    st.write("""
    ### Please enter the following details to check loan approval probability
    """)
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=5)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        income = st.number_input("Annual Income (₹)", min_value=1)
    
    with col2:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=1)
        loan_term = st.number_input("Loan Term (Months)", min_value=1, max_value=360)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
        assets = st.number_input("Total Assets Value (₹)", min_value=0)
    
    # Convert categorical inputs
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    
    if st.button("Predict Loan Approval"):
        # Validate inputs
        if income == 0:
            st.error("Annual Income cannot be zero!")
            return
            
        if loan_amount == 0:
            st.error("Loan Amount cannot be zero!")
            return
            
        # Create input dataframe
        input_data = pd.DataFrame(
            [[dependents, education, self_employed, income, loan_amount, loan_term, cibil_score, assets]],
            columns=[
                "no_of_dependents",
                "education",
                "self_employed",
                "income_annum",
                "loan_amount",
                "loan_term",
                "cibil_score",
                "Assets",
            ],
        )
        
        try:
            # Get prediction
            prediction, probability = predict_loan_approval(input_data)
            
            # Show results
            st.write("---")
            if prediction == 1:
                st.success(f"Loan Approval Probability: {probability:.2%}")
            else:
                st.error(f"Loan Rejection Probability: {(1-probability):.2%}")
            
            # Show risk factors
            st.write("### Key Factors Affecting Decision:")
            
            # Check CIBIL Score
            if cibil_score < 750:
                st.warning("Low CIBIL Score - This might affect your loan approval")
            
            # Check Debt-to-Income ratio (with validation)
            if income > 0:
                dti = loan_amount / income
                if dti > 0.4:
                    st.warning("High Debt-to-Income Ratio - This might affect your loan approval")
            
            # Check Asset Coverage
            if assets < loan_amount:
                st.warning("Low Asset Coverage - This might affect your loan approval")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()
