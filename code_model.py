import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df = pd.read_csv("air.csv")

X = df[
    [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "AQI"
    ]
]
y = df["Health_Risk"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(
    (model, encoder),
    "model.pkl"
)

print("Model saved successfully!")