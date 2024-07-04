from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


app = Flask(__name__)


# Load the model
model = joblib.load('diabetic_patients_readmission_model.pkl')


# Specify the top features
top_features = [
    'number_inpatient', 'number_emergency', 'number_diagnoses',
    'number_outpatient', 'diag_1_428', 'diabetesMed_Yes',
    'num_medications', 'time_in_hospital'
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict(flat=True)

    # Convert string values to float and handle conversion errors
    try:
        data_converted = {key: float(value) for key, value in data.items()}
    except ValueError:
        error_message = "Please enter valid numeric values."
        return render_template('index.html', error=error_message)

    # Create DataFrame from converted data
    df = pd.DataFrame([data_converted], columns=top_features)

    # Check for negative values
    if (df[top_features] < 0).any().any():
        error_message = "Please enter non-negative values only."
        return render_template('index.html', error=error_message)

    # Replace NaN values if any
    df.fillna(0, inplace=True)

    prediction = model.predict(df)
    prediction_value = prediction[0].item()

    return render_template('index.html', prediction=prediction_value)


if __name__ == '__main__':
    app.run(debug=True)
