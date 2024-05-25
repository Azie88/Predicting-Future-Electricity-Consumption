import pandas as pd
import streamlit as st
import numpy as np
import sklearn
import joblib


pipeline = joblib.load('toolkit/Pipeline.joblib')
model = joblib.load('toolkit/model.joblib')

# Add a title and subtitle
st.write("<center><h1>Future Electrification App</h1></center>", unsafe_allow_html=True)

# Set up the layout
col1, col2, col3 = st.columns([1, 3, 3])

st.write("This app uses machine learning to predict acess to electricity based on certain input parameters. Simply enter the required information and click 'Predict' to get a prediction!")
st.subheader("Enter the details to predict access to electricity by % of population")

# Create the input fields
input_data = {}
col1,col2 = st.columns(2)
with col1:
    input_data['Year'] = st.number_input("Input Year",min_value=1990, max_value=2030, step=1)
    input_data['IncomeGroup'] = st.radio("Pick an income group",["Lower middle income", "Upper middle income", "High income", "Low income"])
    input_data['Population'] = st.slider("Population of the country",min_value=9000, max_value=10000000000)

with col2:
    input_data['GDP_per_capita_USD'] = st.slider("Enter GDP per capita (Current $)",min_value=20.0, max_value=250000.0, step=0.1)
    input_data['Inflation_annual_percent'] = st.slider("Enter Consumer Price Index",min_value=-20.0, max_value=25000.0, step=0.1)
    input_data['Consumption (kWh per capita)'] = st.slider("Enter Electricity Consumption",min_value=10.0, max_value=55000.0, step=0.1)

# Define the custom CSS
predict_button_css = """
    <style>
    .predict-button {
        background-color: #C4C4C4;
        color: gray;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
"""

# Display the custom CSS
st.markdown(predict_button_css, unsafe_allow_html=True)

 # Create a button to make a prediction

if st.button("Predict", key="predict_button", help="Click to make a prediction."):
    # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data])


# Selecting categorical and numerical columns separately
        numerical = input_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = input_df.select_dtypes(exclude=[np.number]).columns.tolist()

# Fit and transform the data
        X_processed = pipeline.fit_transform(input_df)

# Extracting feature names for numerical columns
        num_feature_names = numerical

# Extracting feature names for categorical columns after one-hot encoding
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        at_feature_names = cat_encoder.get_feature_names_out(categorical)

# Concatenating numerical and categorical feature names
        feature_names = num_feature_names + list(categorical)

# Convert X_processed to DataFrame
        final_df = pd.DataFrame(X_processed, columns=feature_names)
# Make a prediction
        prediction = model.predict(final_df)[0]


# Display the prediction
        st.write(f"The predicted Access to electricity is: {prediction}.")
        input_df.to_csv("data/input_data.csv", index=False)
        st.table(input_df)