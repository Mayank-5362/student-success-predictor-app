import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define feature descriptions and validation rules
feature_info = {
    "age": {"desc": "Student's age", "min": 10, "max": 30},
    "Medu": {"desc": "Mother's education (0=none, 4=higher)", "min": 0, "max": 4},
    "Fedu": {"desc": "Father's education (0=none, 4=higher)", "min": 0, "max": 4},
    "traveltime": {"desc": "Travel time to school (1=least, 4=most)", "min": 1, "max": 4},
    "freetime": {"desc": "Free time after school (1-5)", "min": 1, "max": 5},
    "health": {"desc": "Health status (1=poor, 5=excellent)", "min": 1, "max": 5}
}

# Load and preprocess the dataset
def load_and_preprocess(file_path, feature_info):
    data = pd.read_csv(file_path)

    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Convert 'passed' to binary
    if 'passed_yes' in data_encoded.columns:
        data_encoded['passed'] = data_encoded['passed_yes']
        data_encoded.drop(columns=['passed_yes'], inplace=True)

    # Keep only relevant columns based on feature_info
    relevant_features = list(feature_info.keys())
    data_encoded = data_encoded[relevant_features + ['passed']]  # Only keep relevant columns

    return data_encoded


# Train the logistic regression model
def train_model(file_path, feature_info):
    data = load_and_preprocess(file_path, feature_info)
    X = data.drop(columns=['passed'])
    y = data['passed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='liblinear', C=1, penalty='l2')
    model.fit(X_train, y_train)

    # Feature importances from the model
    feature_importance = model.coef_[0]  # Coefficients for each feature

    return model, scaler, X.columns, feature_importance


# Predict student success
def predict(model, scaler, feature_columns, form_data):
    student_data = {feature: float(form_data.get(feature, 0)) for feature in feature_columns}
    student_df = pd.DataFrame([student_data])
    student_scaled = scaler.transform(student_df)

    prediction = model.predict(student_scaled)
    return "PASS" if prediction[0] == 1 else "FAIL"
