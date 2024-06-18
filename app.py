import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
import joblib

# Define a function to convert dictionary to DataFrame and prepare for prediction
def dicti_vals(new_person):
    # Convert dictionary to DataFrame
    df = pd.DataFrame([new_person])
    
    # Ensure the DataFrame is in the correct order if necessary (e.g., matching model input order)
    # Assuming the order in `new_person` matches the model's expected input
    return df.values

# Load the pre-trained model using joblib
# try:
#     model = joblib.load('modelj.pkl')
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
#     st.stop()

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# Define the Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # Create form inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=21)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=324)
    bp_systolic = st.number_input("BP Systolic", min_value=0, max_value=300, value=174)
    bp_diastolic = st.number_input("BP Diastolic", min_value=0, max_value=200, value=99)
    heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, value=72)
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    family_history = st.selectbox("Family History", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    exercise_hours = st.number_input("Exercise Hours Per Week", min_value=0.0, max_value=168.0, value=2.07)
    prev_heart_problems = st.selectbox("Previous Heart Problems", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    medication_use = st.selectbox("Medication Use", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=28.17)
    triglycerides = st.number_input("Triglycerides", min_value=0, max_value=1000, value=587)
    sleep_hours = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=4.0)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    diet = st.selectbox("Diet", [0, 1], format_func=lambda x: 'Healthy' if x == 1 else 'Unhealthy')

    # Prediction button
    if st.button("Predict"):
        try:
            # Create a dictionary of inputs
            new_person = {
                'Age': age,
                'Cholesterol': cholesterol,
                'BP_systolic': bp_systolic,
                'BP_diastolic': bp_diastolic,
                'Heart Rate': heart_rate,
                'Diabetes': diabetes,
                'Family History': family_history,
                'Smoking': smoking,
                'Obesity': obesity,
                'Alcohol Consumption': alcohol_consumption,
                'Exercise Hours Per Week': exercise_hours,
                'Previous Heart Problems': prev_heart_problems,
                'Medication Use': medication_use,
                'BMI': bmi,
                'Triglycerides': triglycerides,
                'Sleep Hours Per Day': sleep_hours,
                'Sex': sex,
                'Diet': diet
            }
            
            # Convert dictionary to values array for prediction
            features = dicti_vals(new_person)

            # Make the prediction
            prediction = model.predict(features)

            # Display the prediction
            st.subheader("Prediction Result")
            result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
            st.write(result)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
