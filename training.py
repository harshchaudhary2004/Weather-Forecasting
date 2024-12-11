import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('weatherHistory.csv')


# Preprocessing
df = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Summary']]  # Select relevant features
label_encoder = LabelEncoder()
df['Summary'] = label_encoder.fit_transform(df['Summary'])

# Feature and target split
X = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
y = df['Summary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(model, 'weather_prediction_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved successfully!")

