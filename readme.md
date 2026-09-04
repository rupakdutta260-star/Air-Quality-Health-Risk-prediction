it# Air Quality Health Risk Prediction
## Project Description
This project uses **Machine Learning** and **Python** to predict the **Health Risk** based on different air quality parameters.
The model is trained using the **Random Forest Classifier** algorithm.
## Dataset
The project uses a CSV file named:
`air.csv`
### Input Features
* PM2.5
* PM10
* NO2
* SO2
* CO
* O3
* AQI
### Target
* Health_Risk
## Technologies Used
* Python
* Pandas
* Scikit-learn
* Joblib
## Machine Learning Algorithm

**Random Forest Classifier**
The dataset is divided into:
* **80% Training Data**
* **20% Testing Data**
The model predicts the health risk and calculates the accuracy.
## How the Model Works

1. The program reads data from `air.csv`.
2. Air quality columns are selected as input features.
3. `Health_Risk` is selected as the target.
4. `LabelEncoder` converts health risk labels into numerical values.
5. The data is split into training and testing data.
6. The Random Forest model is trained.
7. The model predicts the health risk using test data.
8. Accuracy is calculated.
9. The trained model and encoder are saved in `model.pkl`.
## Done by
**Rupak Dutta**
BCA Student | Data Science
