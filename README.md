# CropVision


### Crop Yield Prediction and Visualization

This is a **Streamlit** application that predicts crop yields using historical data and visualizes agricultural trends. It provides a user-friendly interface to assist farmers, researchers, and policymakers in making informed decisions based on data-driven insights.

## Features

### 1. Crop Yield Prediction
- Uses **Polynomial Regression** to predict future crop yields based on historical data.
- Allows users to filter predictions by:
  - State
  - Crop
  - Season
- Customize polynomial degree and prediction year range for better accuracy and insights.

### 2. Data Visualization
- **Crop-Wise Average Yield**: Bar charts displaying average yield per crop.
- **Yield Over the Years**: Line plots showing historical yield trends.
- Interactive visualizations using Matplotlib and Seaborn for easy understanding.

### 3. About and Contact Sections
- Explains the purpose, goals, and vision of the project.
- Provides contact details for feedback or queries.

### 4. Interactive Dashboard
- Sidebar navigation to access different sections:
  - Home
  - Visualizations
  - Predictions
  - About Us
  - Contact Us

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- **Python 3.7 or higher**
- Necessary Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Steps to Run the Application
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/crop-yield-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd crop-yield-prediction
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the dataset `crop_yield_dataset.csv` is in the project directory.
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
6. Open the provided URL in your web browser to access the app.

---

## Dataset
The application uses a CSV file, `crop_yield_dataset.csv`, containing the following columns:
- **State**: Name of the state.
- **Crop**: Name of the crop.
- **Season**: Agricultural season (e.g., Kharif, Rabi).
- **Crop_Year**: Year of yield data.
- **Yield**: Crop yield in tons/hectare.

Ensure the dataset is formatted correctly and placed in the project directory before running the app.

---

## Usage

1. **Home**:
   - View a brief introduction and a preview of the dataset.

2. **Visualizations**:
   - Explore interactive charts and graphs to understand historical trends.

3. **Predictions**:
   - Select the state, crop, and season to predict yields.
   - Customize the polynomial degree and prediction year range.
   - View predicted yields and trend lines.

4. **About Us**:
   - Learn about the project objectives, vision, and future enhancements.

5. **Contact Us**:
   - Find details to reach out for queries or feedback.

---

## Future Enhancements
- Integration of weather data for better predictions.
- Inclusion of real-time data updates.
- Mobile-friendly interface.
- Analysis of soil quality, pest control, and water usage.

---

## Contact Us
For any queries or feedback, feel free to reach out:

- **Email**: info@cropprediction.com
- **Phone**: +123-456-7890
- **Address**: 123 Agri Street, Farm City, AgriLand

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
We extend our gratitude to:
- The **open-source community** for providing the tools and frameworks used.
- **Agricultural departments and researchers** for valuable insights into crop yield trends.
