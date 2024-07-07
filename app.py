from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load('Project.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about_project')
def about_project():
    return render_template('about_project.html')

@app.route('/about_developer')
def about_developer():
    return render_template('about_developer.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract and convert data from form
            features = {
                'CRIM': float(request.form['CRIM']),
                'ZN': float(request.form['ZN']),
                'INDUS': float(request.form['INDUS']),
                'CHAS': float(request.form['CHAS']),
                'NOX': float(request.form['NOX']),
                'RM': float(request.form['RM']),
                'AGE': float(request.form['AGE']),
                'DIS': float(request.form['DIS']),
                'RAD': float(request.form['RAD']),
                'TAX': float(request.form['TAX']),
                'PTRATIO': float(request.form['PTRATIO']),
                'B': float(request.form['B']),
                'LSTAT': float(request.form['LSTAT']),
            }
            
            # Convert data into a DataFrame
            data = pd.DataFrame([features])
            # Predict the price
            prediction = model.predict(data)
            print(prediction)
            # Render prediction result
            return render_template('home.html', prediction_text = "Predicted MEDV: " + str(prediction[0]))
        except Exception as e:
            return render_template('home.html', prediction_text=f'Error: Enter the values correctly')

if __name__ == '__main__':
    app.run(debug=True)
