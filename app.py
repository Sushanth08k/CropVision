import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
def load_data():
    dataset_path = "crop_yield_dataset.csv"  # Ensure the dataset is in the same folder
    try:
        data = pd.read_csv(dataset_path)
        return data
    except FileNotFoundError:
        st.error("Dataset file 'crop_yield_dataset.csv' not found in the current folder.")
        return None

# Polynomial Regression for Prediction
def polynomial_regression(data, selected_state, selected_crop, selected_season):
    st.write("### Training Polynomial Regression Model for Yield Prediction")

    # Filter data by selected state, crop, and season
    filtered_data = data[ 
        (data["State"] == selected_state) & 
        (data["Crop"] == selected_crop) & 
        (data["Season"] == selected_season)
    ]

    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return

    # Extract features and target
    X = filtered_data["Crop_Year"].values.reshape(-1, 1)
    y = filtered_data["Yield"].values

    # Polynomial feature transformation
    degree = st.slider("Select Polynomial Degree for Trend Model", 1, 5, 2)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    # Predict yields for a range of years
    year_range = st.slider("Select Year Range for Prediction", int(X.min()), int(X.max()) + 10, (int(X.min()), int(X.max())))
    year_range_array = np.arange(year_range[0], year_range[1] + 1).reshape(-1, 1)
    predicted_yields = model.predict(poly.transform(year_range_array))

    st.write("### Predicted Yields")
    prediction_results = pd.DataFrame({
        "Year": year_range_array.flatten(),
        "Predicted Yield": predicted_yields
    })
    st.write(prediction_results)

    # Visualize historical data and trend
    st.write("### Historical Data and Trend")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=filtered_data["Crop_Year"], y=filtered_data["Yield"], label="Historical Data", color="blue")
    plt.plot(year_range_array, predicted_yields, color="red", label="Trend Line")
    plt.xlabel("Year")
    plt.ylabel("Yield")
    plt.legend()
    st.pyplot(plt.gcf())

# Data Visualization
def visualize_data(data):
    st.write("### Data Visualizations")

    # Crop-wise average yield
    st.write("#### Average Yield by Crop")
    crop_yield = data.groupby("Crop")["Yield"].mean().sort_values()
    plt.figure(figsize=(10, 6))
    crop_yield.plot(kind="bar", color="teal")
    plt.title("Average Yield by Crop")
    plt.xlabel("Crop")
    plt.ylabel("Average Yield")
    st.pyplot(plt.gcf())

    # Yield over years
    st.write("#### Yield Over the Years")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Crop_Year", y="Yield", data=data, marker="o")
    plt.title("Yield Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Yield")
    st.pyplot(plt.gcf())

# About Us Section
def about_us():
    st.write("### About Us")
    st.write("""
    This application, Crop Yield Prediction and Visualization, is a powerful tool designed to assist farmers, agricultural researchers, and policymakers in making data-driven decisions for sustainable farming. By leveraging historical data and machine learning models, this tool provides accurate crop yield predictions and insights into agricultural trends.
             
             
        Objectives of the Project
Empower Farmers: Enable farmers to make informed decisions about crop planning based on expected yields and historical trends.
             
Support Agricultural Policies:
Assist policymakers in framing effective policies and resource allocation by providing regional and seasonal insights.
Promote Data-Driven Agriculture: Encourage the use of advanced analytics and technology in agriculture to increase productivity and reduce risks.
Features of the Application
             
Crop Yield Prediction:
 Predict future crop yields using polynomial regression models.
Historical Trends Visualization: Understand past crop yield trends across states, seasons, and crop types.
Interactive Dashboard: Filter data based on specific crops, states, or seasons for tailored insights.
User-Friendly Interface: Simplified design for ease of use, even for non-technical users.
Benefits of Using This Tool
             
Improved Planning:
Optimize the selection of crops and resources for better yield outcomes.
Risk Mitigation: Predict yield fluctuations and adjust plans accordingly.
Research Insights: Facilitate in-depth research on agricultural trends to improve farming practices.
             
    Future Enhancements
We aim to continuously improve and expand the functionality of this application. Future updates may include:

Incorporating weather forecasts and other external factors into prediction models.
Adding real-time data updates for more accurate predictions.
Expanding the scope to include insights on soil quality, pest control, and water usage.
Enabling mobile compatibility for on-the-go access.
             

    Our Vision
We envision a future where agriculture is seamlessly integrated with cutting-edge technology to address global challenges such as food security, climate change, and resource optimization. Our mission is to contribute to this vision by creating tools that simplify complex problems and empower stakeholders in the agricultural ecosystem.


    """)

# Contact Us Section
def contact_us():
    st.write("### Contact Us")
    st.write("""
    For any queries or feedback, feel free to reach out to us:
    
    - Email: ksushanth9030@gmail.com
    - Phone: +91 9392361766
    - Address: LB Nagar, Hyderabad, India
    """)

# Main Function
def main():
    st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
    st.title("Crop Yield Prediction and Visualization")
    st.markdown(
        "This application allows you to predict crop yields based on historical data and visualize trends."
    )

    # Load data
    data = load_data()

    if data is not None:
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        section = st.sidebar.radio("Go to", ["Home", "About Us", "Contact", "Visualizations", "Predictions"])

        if section == "Home":
            st.write("### Welcome to the Crop Yield Prediction Tool")
            st.write("This tool helps to predict and visualize crop yields based on historical data.")
            st.write("### Data Preview")
            st.write(data.head())

        elif section == "About Us":
            about_us()

        elif section == "Contact":
            contact_us()

        elif section == "Visualizations":
            visualize_data(data)

        elif section == "Predictions":
            # Filters for prediction
            states = data["State"].unique()
            crops = data["Crop"].unique()
            seasons = data["Season"].unique()

            selected_state = st.selectbox("Select State", states)
            selected_crop = st.selectbox("Select Crop", crops)
            selected_season = st.selectbox("Select Season", seasons)

            polynomial_regression(data, selected_state, selected_crop, selected_season)

    else:
        st.info("Please ensure the dataset file 'crop_yield_dataset.csv' is in the same folder as this script.")

if __name__ == "__main__":
    main()
