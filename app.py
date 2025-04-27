# File: app.py
from flask import Flask, request, render_template
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load model and label encoder
model = joblib.load('childhood_disease_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load dataset to get symptom names
data = pd.read_csv('Filtered_Childhood_Dataset.csv')
symptoms = data.columns[:-1].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    selected_symptoms = []
    
    if request.method == 'POST':
        # Get selected symptoms
        selected_symptoms = request.form.getlist('symptoms')
        
        
        input_data = pd.DataFrame(0, index=[0], columns=symptoms)
        for symptom in selected_symptoms:
            if symptom in input_data.columns:
                input_data[symptom] = 1
        
        # Predict
        prediction = model.predict(input_data)
        disease = le.inverse_transform(prediction)[0]
        result = {
            'disease': disease,
            'symptoms': selected_symptoms
        }
    
    return render_template('index.html', symptoms=symptoms, result=result)

if __name__ == '__main__':
    app.run(debug=True)