from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load model
model = joblib.load('models/best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:

        gender = float(request.form.get('gender', 0))
        age = float(request.form.get('age', 0))
        flight_distance = float(request.form.get('flight_distance', 0))
        wifi = float(request.form.get('wifi', 0))
        online_boarding = float(request.form.get('online_boarding', 0))

        data = np.array([[
            gender,
            age,
            flight_distance,
            wifi,
            online_boarding
        ]])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "PENUMPANG PUAS"
        else:
            result = "NEUTRAL/BIASA SAJA"

        return render_template(
            'index.html',
            prediction_text=f'Hasil Prediksi: {result}'
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'ERROR: {str(e)}'
        )

if __name__ == "__main__":
    app.run(debug=True)