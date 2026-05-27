from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/student_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    sleep_hours = float(request.form['sleep_hours'])
    previous_score = float(request.form['previous_score'])
    assignments_completed = float(request.form['assignments_completed'])

    features = np.array([[
        study_hours,
        attendance,
        sleep_hours,
        previous_score,
        assignments_completed
    ]])

    prediction = model.predict(features)[0]

    if prediction >= 85:
        grade = "A"
    elif prediction >= 70:
        grade = "B"
    elif prediction >= 50:
        grade = "C"
    else:
        grade = "Fail"

    return render_template(
        'result.html',
        prediction=round(prediction, 2),
        grade=grade
    )


if __name__ == '__main__':
    app.run(debug=True)