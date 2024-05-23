import pandas as pd
import streamlit as st
import numpy as np
import sklearn
import joblib


encoder = joblib.load('toolkit/Encoder.joblib')
scaler = joblib.load('toolkit/Scaler.joblib')
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
    input_data['Country/Region'] = st.selectbox("Choose Country/Region", [
        'Afghanistan', 'Africa Eastern and Southern', 'Africa Western and Central', 'Albania', 'Algeria', 
        'American Samoa', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Arab World', 'Argentina', 'Armenia', 
        'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The', 'Bahrain', 'Bangladesh', 'Barbados', 
        'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 
        'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 
        'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Caribbean small states', 'Cayman Islands', 
        'Central African Republic', 'Central Europe and the Baltics', 'Chad', 'Channel Islands', 'Chile', 
        'China', 'Colombia', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Costa Rica', "Cote d'Ivoire", 
        'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czechia', 'Denmark', 'Djibouti', 'Dominica', 
        'Dominican Republic', 'Ecuador', 'Egypt, Arab Rep.', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 
        'Estonia', 'Eswatini', 'Ethiopia', 'Euro area', 'Europe & Central Asia', 
        'Europe & Central Asia (excluding high income)', 'Europe & Central Asia (IDA & IBRD countries)', 
        'European Union', 'Faroe Islands', 'Fiji', 'Finland', 'Fragile and conflict affected situations', 
        'France', 'French Polynesia', 'Gabon', 'Gambia, The', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 
        'Greece', 'Greenland', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 
        'Heavily indebted poor countries (HIPC)', 'Honduras', 'Hong Kong SAR, China', 'Hungary', 'IBRD only', 
        'Iceland', 'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total', 'India', 'Indonesia', 'Iran, Islamic Rep.', 
        'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 
        'Kenya', 'Kiribati', 'Korea, Dem. People\'s Rep.', 'Korea, Rep.', 'Kosovo', 'Kuwait', 'Kyrgyz Republic', 
        'Lao PDR', 'Late-demographic dividend', 'Latvia', 'Least developed countries: UN classification', 
        'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Low & middle income', 
        'Low income', 'Lower middle income', 'Luxembourg', 'Macao SAR, China', 'Madagascar', 'Malawi', 
        'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 
        'Micronesia, Fed. Sts.', 'Middle East & North Africa', 'Middle East & North Africa (excluding high income)', 
        'Middle East & North Africa (IDA & IBRD countries)', 'Middle income', 'Moldova', 'Monaco', 'Mongolia', 
        'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 
        'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North America', 'North Macedonia', 
        'Northern Mariana Islands', 'Norway', 'OECD members', 'Oman', 'Other small states', 
        'Pacific island small states', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 
        'Philippines', 'Poland', 'Portugal', 'Post-demographic dividend', 'Pre-demographic dividend', 'Puerto Rico', 
        'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'Samoa', 'San Marino', 'Sao Tome and Principe', 
        'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten (Dutch part)', 
        'Slovak Republic', 'Slovenia', 'Small states', 'Solomon Islands', 'Somalia', 'South Africa', 'South Asia', 
        'South Asia (IDA & IBRD)', 'South Sudan', 'Spain', 'Sri Lanka', 'St. Kitts and Nevis', 'St. Lucia', 
        'St. Martin (French part)', 'St. Vincent and the Grenadines', 'Sub-Saharan Africa', 
        'Sub-Saharan Africa (excluding high income)', 'Sub-Saharan Africa (IDA & IBRD countries)', 
        'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Tanzania', 
        'Thailand', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkiye', 'Turkmenistan', 
        'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 
        'United States', 'Upper middle income', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela, RB', 
        'Viet Nam', 'Virgin Islands (U.S.)', 'West Bank and Gaza', 'World', 'Yemen, Rep.', 'Zambia', 'Zimbabwe'
        ])
    input_data['Population'] = st.slider("Population of the country/region",min_value=9000, max_value=10000000000, step=1)
    input_data['GDP per capita (current US$)'] = st.slider("Enter GDP per capita",min_value=20.0, max_value=250000.0, step=0.1)

with col2:
    input_data['Inflation (annual %)'] = st.slider("Enter Consumer Price Index",min_value=-20.0, max_value=25000.0, step=0.1)
    input_data['Consumption (kWh per capita)'] = st.number_input("Enter Electricity Consumption",min_value=10.0, max_value=55000.0, step=0.1)
    input_data['Year'] = st.slider("Choose Year",min_value=1990, max_value=2030, step=1)

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
        cat_columns = ['Country/Region']
        num_columns = ['Population', 'GDP per capita (current US$)', 'Inflation (annual %)', 'Consumption (kWh per capita)', 'Year_Col']

# Scale the numerical columns
        input_df_scaled = scaler.transform(num_columns)
        input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)


# Encode the categorical columns
        input_encoded_df = encoder.transform(input_df[cat_columns]).toarray()
        input_encoded_columns = encoder.get_feature_names_out(cat_columns)
        input_encoded_df = pd.DataFrame(input_encoded_df, columns=input_encoded_columns)
 

#joining the cat encoded and num scaled
        final_df = pd.concat([input_scaled_df, input_encoded_df], axis=1)

# Make a prediction
        prediction = model.predict(final_df)[0]


# Display the prediction
        st.write(f"The predicted sales are: {prediction}.")
        input_df.to_csv("data/input_data.csv", index=False)
        st.table(input_df)