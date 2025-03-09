from flask import Flask, render_template, request
from models.student_success import train_model, predict

app = Flask(__name__)

# Define feature descriptions and validation rules
feature_info = {
    "age": {"desc": "Student's age", "min": 10, "max": 30},
    "Medu": {"desc": "Mother's education (0=none, 4=higher)", "min": 0, "max": 4},
    "Fedu": {"desc": "Father's education (0=none, 4=higher)", "min": 0, "max": 4},
    "traveltime": {"desc": "Travel time to school (1=least, 4=most)", "min": 1, "max": 4},
    "freetime": {"desc": "Free time after school (1-5)", "min": 1, "max": 5},
    "health": {"desc": "Health status (1=poor, 5=excellent)", "min": 1, "max": 5}
}

file_path = "data/student-data.csv"  # Update with the correct path if necessary

# Train the model
model, scaler, feature_columns, feature_importance = train_model(file_path, feature_info)

@app.route('/')
def index():
    return render_template('index.html', features=feature_columns, feature_info=feature_info)

@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        # Validate inputs
        errors = {}
        for feature, info in feature_info.items():
            value = request.form.get(feature)
            try:
                value = float(value)
                if value < info["min"] or value > info["max"]:
                    errors[feature] = f"Value must be between {info['min']} and {info['max']}."
            except ValueError:
                errors[feature] = "Invalid number."

        if errors:
            return render_template('index.html', features=feature_columns, feature_info=feature_info, errors=errors)

        # Make prediction
        prediction = predict(model, scaler, feature_columns, request.form)

        # Pass feature importance as JSON
        feature_importance_data = {k: float(v) for k, v in zip(feature_columns, feature_importance)}

        return render_template('result.html', prediction=prediction, feature_importance=feature_importance_data)

if __name__ == '__main__':
    app.run(debug=True)
